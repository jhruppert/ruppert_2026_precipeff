"""Per-timestep SAM RCE class diagnostics for the mean-profiles violin figure.

The product deliberately keeps one record per CTL model time. A daily NetCDF
file therefore contains one independent plotting observation per input record
without retaining the large raw horizontal fields.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import xarray as xr

from dependencies.rce_binned_cross import (
    CLASSIFIER_VERSION,
    DEFAULT_CTL_NAME,
    DEFAULT_DATA_ROOT,
    PCLASS_FILL_VALUE,
    PRESSURE_HPA,
    RCEProcessingError,
    PreprocessConfig,
    TimeIndexed2DFields,
    _numeric_encoding,
    _preflight,
    _squeezed_array,
    _time_key,
    align_vertical_velocity,
    atomic_to_netcdf,
    density_from_state,
    generate_pclass,
    interpolate_fields_to_pressure,
    load_pclass_sidecar,
    mixing_ratio_from_dataset,
    pclass_cache_path,
    pressure_from_dataset,
    validate_lw_acre_inputs,
    validate_pclass,
    window_tag,
    write_pclass_sidecar,
    xy_coordinates,
    group_records_by_day,
)


VIOLIN_SCHEMA_VERSION = "3.0"
VIOLIN_CLASS_CODES = np.arange(7, dtype=np.int8)
VIOLIN_CLASS_NAMES = np.array(
    ["Non-cloud", "Deep", "Congestus", "Shallow", "Stratiform", "Anvil", "DSA"]
)
PLOT_CLASS_CODES = np.array([1, 4, 5, 6], dtype=np.int8)
LV = 2.5e6
SECONDS_PER_DAY = 86400.0


def violin_daily_cache_path(cache_dir: Path | str, day: int) -> Path:
    return Path(cache_dir) / "violin_daily" / f"rce_violin_day_{int(day):04d}.nc"


def violin_output_path(output_dir: Path | str, start_day: int, end_day: int) -> Path:
    return Path(output_dir) / f"rce_ctl_violin_{window_tag(start_day, end_day)}.nc"


def _class_masks(pclass: np.ndarray) -> list[np.ndarray]:
    values = np.asarray(pclass)
    valid = values != PCLASS_FILL_VALUE
    masks = [(valid & (values == code)) for code in range(6)]
    masks.append(valid & np.isin(values, [1, 4, 5]))
    return masks


def _class_sums(values: np.ndarray, pclass: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return class-conditioned sums and finite-value counts."""
    values = np.asarray(values, dtype=float)
    masks = _class_masks(pclass)
    sums = np.zeros((7,) + values.shape[:-2], dtype=float)
    counts = np.zeros((7,) + values.shape[:-2], dtype=np.int64)
    for iclass, mask in enumerate(masks):
        expanded = mask
        while expanded.ndim < values.ndim:
            expanded = expanded[None, ...]
        finite = expanded & np.isfinite(values)
        sums[iclass] = np.where(finite, values, 0.0).reshape(values.shape[:-2] + (-1,)).sum(-1)
        counts[iclass] = finite.reshape(values.shape[:-2] + (-1,)).sum(-1)
    return sums, counts


def _scalar_class_sums(values: np.ndarray, pclass: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=float)
    masks = _class_masks(pclass)
    sums = np.zeros(7, dtype=float)
    counts = np.zeros(7, dtype=np.int64)
    for iclass, mask in enumerate(masks):
        finite = mask & np.isfinite(values)
        sums[iclass] = np.nansum(np.where(finite, values, np.nan))
        counts[iclass] = np.count_nonzero(finite)
    return sums, counts


def _daily_dataset(
    day: int,
    times: Sequence[float],
    class_count: np.ndarray,
    domain_count: np.ndarray,
    rain_sum: np.ndarray,
    rain_count: np.ndarray,
    acre_sum: np.ndarray,
    acre_count: np.ndarray,
    up_sum: np.ndarray,
    up_count: np.ndarray,
    down_sum: np.ndarray,
    down_count: np.ndarray,
    records,
    attrs: Mapping[str, object],
) -> xr.Dataset:
    ds = xr.Dataset(
        {
            "class_count": (("time", "class_code"), class_count),
            "domain_count": (("time",), domain_count),
            "rain_sum": (("time", "class_code"), rain_sum),
            "rain_count": (("time", "class_code"), rain_count),
            "lw_acre_sum": (("time", "class_code"), acre_sum),
            "lw_acre_count": (("time", "class_code"), acre_count),
            "up_mass_flux_sum": (("time", "pressure_hpa", "class_code"), up_sum),
            "up_mass_flux_count": (("time", "pressure_hpa", "class_code"), up_count),
            "down_mass_flux_sum": (("time", "pressure_hpa", "class_code"), down_sum),
            "down_mass_flux_count": (("time", "pressure_hpa", "class_code"), down_count),
        },
        coords={
            "time": np.asarray(times, dtype=float),
            "pressure_hpa": PRESSURE_HPA,
            "class_code": VIOLIN_CLASS_CODES,
            "class_name": (("class_code",), VIOLIN_CLASS_NAMES),
        },
        attrs={
            "schema_version": VIOLIN_SCHEMA_VERSION,
            "day": int(day),
            "time_bounds": f"[{day}, {day + 1})",
            "sample_count": len(times),
            "sample_times": json.dumps([float(t) for t in times]),
            "ctl_source_files": json.dumps([str(r.path) for r in records]),
            "classifier_version": CLASSIFIER_VERSION,
            "class_definition": "codes 0-5 primitive SAM classes; code 6 = 1+4+5 DSA",
            "area_fraction_definition": "100 * class_count / domain_count",
            "lw_acre_definition": "(LWNS-LWNT)-(LWNSC-LWNTC)",
            "lw_acre_interpretation": "local CTL all-sky minus clear-sky atmospheric LW flux convergence",
            "gef_definition": "LW-ACRE / (Lv * CTL class precipitation)",
            "epsilon_definition": "1 - (-Md / Mu)",
            **dict(attrs),
        },
    )
    ds["lw_acre_sum"].attrs["units"] = "W m-2 summed over columns"
    for name in ("up_mass_flux_sum", "down_mass_flux_sum"):
        ds[name].attrs["units"] = "kg m-2 s-1 summed over columns"
    ds["rain_sum"].attrs["units"] = "mm day-1 summed over columns"
    return ds


def preprocess_violin_caches(config: PreprocessConfig) -> list[Path]:
    """Build missing per-timestep/class daily caches."""
    config.validate()
    matched, _, inventory = _preflight(config, require_lw_heating=False)
    if config.inspect_only:
        inventory["lw_acre_flux_units"] = validate_lw_acre_inputs(config, matched)
        print(json.dumps(inventory, indent=2))
        return []
    two_d_index = TimeIndexed2DFields(
        config.ctl_root, config.start_day, config.end_day, config.time_tolerance
    )
    legacy = __import__("rce_binned_cross").LegacyDiagnosticsCache(config.pickle_dir, config.time_tolerance)
    written = []
    for day, records in group_records_by_day(matched, config.start_day, config.end_day).items():
        output = violin_daily_cache_path(config.cache_dir, day)
        if output.exists() and not config.overwrite:
            with xr.open_dataset(output) as existing:
                if existing.attrs.get("schema_version") != VIOLIN_SCHEMA_VERSION or int(existing.attrs.get("day", -1)) != day:
                    raise RCEProcessingError(f"Incompatible violin cache: {output}")
            written.append(output)
            continue
        ntime = len(records)
        npressure = len(PRESSURE_HPA)
        shape7 = (ntime, 7)
        shape7p = (ntime, npressure, 7)
        class_count = np.zeros(shape7, dtype=np.int64)
        domain_count = np.zeros(ntime, dtype=np.int64)
        rain_sum = np.zeros(shape7)
        rain_count = np.zeros(shape7, dtype=np.int64)
        acre_sum = np.zeros(shape7); acre_count = np.zeros(shape7, dtype=np.int64)
        up_sum = np.zeros(shape7p); up_count = np.zeros(shape7p, dtype=np.int64)
        down_sum = np.zeros(shape7p); down_count = np.zeros(shape7p, dtype=np.int64)
        pclass_sidecar = pclass_cache_path(config.cache_dir, day)
        x_coord = y_coord = None
        pclasses = {}
        generated = False
        source_map = {_time_key(r.time, config.time_tolerance): r.path for r in records}
        ordered_pclasses = []
        for itime, record in enumerate(records):
            with xr.open_dataset(record.path, decode_times=False) as ctl:
                x, y = xy_coordinates(ctl)
                if x_coord is None:
                    x_coord, y_coord = x, y
                    if pclass_sidecar.exists():
                        pclasses = load_pclass_sidecar(pclass_sidecar, [r.time for r in records], x, y, source_map, config.time_tolerance)
                pressure = pressure_from_dataset(ctl)
                temperature = _squeezed_array(ctl, "TABS", 3)
                qv = mixing_ratio_from_dataset(ctl, "QV")
                if _time_key(record.time, config.time_tolerance) in pclasses:
                    pclass = validate_pclass(pclasses[_time_key(record.time, config.time_tolerance)], pressure.shape[1:])
                else:
                    cached = legacy.get("ctl", "pclass", record.time, pressure.shape[1:], x, y, record.path)
                    if cached is not None:
                        pclass = validate_pclass(cached, pressure.shape[1:])
                    else:
                        pclass = generate_pclass(temperature, mixing_ratio_from_dataset(ctl, "QN"), mixing_ratio_from_dataset(ctl, "QP"), pressure)
                    pclasses[_time_key(record.time, config.time_tolerance)] = pclass
                    generated = True
                ordered_pclasses.append(pclass)
                masks = _class_masks(pclass)
                valid = pclass != PCLASS_FILL_VALUE
                domain_count[itime] = np.count_nonzero(valid)
                class_count[itime] = np.array([np.count_nonzero(m) for m in masks])
                w = align_vertical_velocity(_squeezed_array(ctl, "W", 3), pressure.shape[0])
                rho = density_from_state(pressure, temperature, qv)
                interp = interpolate_fields_to_pressure({"mass": rho * w}, pressure)
                for iclass, mask in enumerate(masks):
                    for name, target_sum, target_count in (("mass", up_sum, up_count),):
                        values = interp[name]
                        if name == "mass": values = np.where(values > 0, values, np.nan)
                        for ilev in range(npressure):
                            finite = mask & np.isfinite(values[ilev])
                            target_sum[itime, ilev, iclass] = np.nansum(np.where(finite, values[ilev], np.nan))
                            target_count[itime, ilev, iclass] = np.count_nonzero(finite)
                    values = np.where(interp["mass"] < 0, interp["mass"], np.nan)
                    for ilev in range(npressure):
                        finite = mask & np.isfinite(values[ilev])
                        down_sum[itime, ilev, iclass] = np.nansum(np.where(finite, values[ilev], np.nan))
                        down_count[itime, ilev, iclass] = np.count_nonzero(finite)
                rain, _, rx, ry = two_d_index.read_rain(
                    record.time, config.rain_var, config.rain_units
                )
                if not np.allclose(x, rx) or not np.allclose(y, ry):
                    raise RCEProcessingError("3-D and precipitation grids differ")
                for iclass, mask in enumerate(masks):
                    finite = mask & np.isfinite(rain)
                    rain_sum[itime, iclass] = np.nansum(np.where(finite, rain, np.nan))
                    rain_count[itime, iclass] = np.count_nonzero(finite)
                acre, _, ax, ay = two_d_index.read_lw_acre(record.time)
                if not np.allclose(x, ax) or not np.allclose(y, ay):
                    raise RCEProcessingError("3-D and LW-ACRE flux grids differ")
                for iclass, mask in enumerate(masks):
                    finite = mask & np.isfinite(acre)
                    acre_sum[itime, iclass] = np.nansum(np.where(finite, acre, np.nan))
                    acre_count[itime, iclass] = np.count_nonzero(finite)
        if generated:
            write_pclass_sidecar(pclass_sidecar, [r.time for r in records], ordered_pclasses, x_coord, y_coord, [r.path for r in records])
        ds = _daily_dataset(
            day, [r.time for r in records], class_count, domain_count, rain_sum,
            rain_count, acre_sum, acre_count, up_sum, up_count, down_sum,
            down_count, records, {}
        )
        atomic_to_netcdf(ds, output, _numeric_encoding(ds))
        written.append(output)
    return written


def aggregate_violin_caches(cache_dir: Path | str, start_day: int = 1, end_day: int = 21) -> xr.Dataset:
    paths = [violin_daily_cache_path(cache_dir, d) for d in range(start_day, end_day)]
    if any(not p.exists() for p in paths):
        raise FileNotFoundError("Missing violin daily cache: " + ", ".join(str(p) for p in paths if not p.exists()))
    datasets = [xr.load_dataset(p) for p in paths]
    try:
        output = xr.concat(datasets, dim="time").sortby("time")
        output.attrs.update({
            "start_day_inclusive": int(start_day), "end_day_exclusive": int(end_day),
            "time_bounds": f"[{start_day}, {end_day})", "schema_version": VIOLIN_SCHEMA_VERSION,
            "source_daily_files": json.dumps([str(p) for p in paths]),
        })
        return output
    finally:
        for ds in datasets: ds.close()


def compute_violin_metrics(dataset: xr.Dataset) -> xr.Dataset:
    """Convert cached sums/counts into per-time/class plotting variables."""
    rain_sum = dataset.rain_sum.values
    rain_count = dataset.rain_count.values
    acre_sum = dataset.lw_acre_sum.values
    acre_count = dataset.lw_acre_count.values
    up_sum = dataset.up_mass_flux_sum.values
    up_count = dataset.up_mass_flux_count.values
    down_sum = dataset.down_mass_flux_sum.values
    down_count = dataset.down_mass_flux_count.values
    rain = np.divide(rain_sum, rain_count, out=np.full_like(rain_sum, np.nan, dtype=float), where=rain_count > 0)
    acre = np.divide(
        acre_sum, acre_count, out=np.full_like(acre_sum, np.nan, dtype=float),
        where=acre_count > 0,
    )
    up = np.divide(up_sum, up_count, out=np.full_like(up_sum, np.nan, dtype=float), where=up_count > 0)
    down = np.divide(down_sum, down_count, out=np.full_like(down_sum, np.nan, dtype=float), where=down_count > 0)
    pressure = dataset.pressure_hpa.values
    order = np.argsort(pressure)
    rain_flux = rain / SECONDS_PER_DAY
    gef = np.divide(acre, LV * rain_flux, out=np.full_like(acre, np.nan), where=np.isfinite(acre) & (rain_flux > 0))
    up_integrated = np.trapz(up[:, order, :], pressure[order] * 100.0, axis=1)
    down_integrated = np.trapz(down[:, order, :], pressure[order] * 100.0, axis=1)
    epsilon = 1.0 - np.divide(-down_integrated, up_integrated, out=np.full_like(up_integrated, np.nan), where=np.isfinite(up_integrated) & (up_integrated > 0) & np.isfinite(down_integrated))
    area = np.divide(100.0 * dataset.class_count.values, dataset.domain_count.values[:, None], out=np.full_like(dataset.class_count.values, np.nan, dtype=float), where=dataset.domain_count.values[:, None] > 0)
    return xr.Dataset({
        "area_fraction": (("time", "class_code"), area),
        "lw_acre": (("time", "class_code"), acre),
        "gef": (("time", "class_code"), gef),
        "precip_mm_day": (("time", "class_code"), rain),
        "epsilon": (("time", "class_code"), epsilon),
        "up_mass_flux": (("time", "pressure_hpa", "class_code"), up),
        "down_mass_flux": (("time", "pressure_hpa", "class_code"), down),
    }, coords=dataset.coords, attrs=dataset.attrs)


def plot_violin(metrics: xr.Dataset, output_dir: Path | str, start_day: int, end_day: int) -> Path:
    import matplotlib.pyplot as plt
    import seaborn as sns
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(7, 3), layout="constrained", dpi=300)
    variables = ["area_fraction", "gef", "epsilon"]
    titles = ["(a) Area fraction", "(b) GEF", r"(c) $\epsilon$"]
    units = ["%", "", ""]
    labels = ["Deep", "Stratiform", "Anvil", "DSA"]
    for iax, (ax, variable, title, unit) in enumerate(zip(axes, variables, titles, units)):
        values = [metrics[variable].sel(class_code=int(code)).values.ravel() for code in PLOT_CLASS_CODES]
        values = [v[np.isfinite(v)] for v in values]
        sns.violinplot(data=values, inner="box", width=0.7, ax=ax)
        ax.set_title(title); ax.set_ylabel(unit); ax.set_xticks([])
        if variable == "gef": ax.set_yscale("symlog", linthresh=0.01)
        if variable == "area_fraction": ax.set_ylim(0, 100)
        if iax < 2: ax.get_legend() and ax.get_legend().remove()
        sns.despine(offset=10, ax=ax, bottom=True)
    axes[-1].legend(labels, loc="upper right", bbox_to_anchor=(2.0, 0.75), frameon=False)
    stem = f"rce_ctl_violin_{window_tag(start_day, end_day)}"
    for suffix in ("png", "pdf"): fig.savefig(output_dir / f"{stem}.{suffix}", bbox_inches="tight")
    plt.close(fig)
    return output_dir / f"{stem}.png"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build SAM RCE class-conditioned violin diagnostics")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--ctl-root", type=Path)
    parser.add_argument("--cache-dir", type=Path); parser.add_argument("--pickle-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("figures"))
    parser.add_argument("--start-day", type=int, default=1); parser.add_argument("--end-day", type=int, default=21)
    parser.add_argument("--lw-var"); parser.add_argument("--lw-units"); parser.add_argument("--rain-var", default="Prec"); parser.add_argument("--rain-units")
    parser.add_argument("--overwrite", action="store_true"); parser.add_argument("--inspect", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv); root = args.data_root
    config = PreprocessConfig(ctl_root=args.ctl_root or root / DEFAULT_CTL_NAME, cache_dir=args.cache_dir or root / "pickle_out" / "binned_cross_2d_rce", pickle_dir=args.pickle_dir or root / "pickle_out", start_day=args.start_day, end_day=args.end_day, lw_var=args.lw_var, lw_units=args.lw_units, rain_var=args.rain_var, rain_units=args.rain_units, overwrite=args.overwrite, inspect_only=args.inspect)
    preprocess_violin_caches(config)
    if not args.inspect:
        aggregate = aggregate_violin_caches(config.cache_dir, args.start_day, args.end_day)
        metrics = compute_violin_metrics(aggregate)
        out = violin_output_path(args.output_dir, args.start_day, args.end_day)
        atomic_to_netcdf(metrics, out, _numeric_encoding(metrics)); plot_violin(metrics, args.output_dir, args.start_day, args.end_day)
        print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    main()
