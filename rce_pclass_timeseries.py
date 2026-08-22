"""Time-resolved SAM RCE diagnostics conditioned on precipitation class.

The workflow streams one single-time SAM 3-D file at a time, reduces the
horizontal fields to class-conditioned diagnostics, and writes small daily
NetCDF caches.  The caches are concatenated into one experiment-aligned time
series without re-reading the raw three-dimensional model output.

The precipitation classifier and SAM input handling deliberately reuse
``rce_binned_cross`` so the output remains compatible with
``precip_eff_rce.ipynb``.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import xarray as xr

import rce_binned_cross as rce
from thermo_functions import rv_saturation


SCHEMA_VERSION = "1.0"
DIAGNOSTIC_VERSION = "sam_pclass_timeseries_v1"

DEFAULT_DATA_ROOT = rce.DEFAULT_DATA_ROOT
DEFAULT_CTL_NAME = rce.DEFAULT_CTL_NAME
DEFAULT_RH_NAME = "RCE_768_305_radhomo"

EXPERIMENT_IDS = ("RCE_CTL", "RCE_RH")
PCLASS_CODES = np.array([1, 4, 5, 6], dtype=np.int8)
PCLASS_NAMES = np.array(["DeepC", "Strat", "Anvil", "DSA"])
PCLASS_MEMBERS = {
    1: (1,),
    4: (4,),
    5: (5,),
    6: (1, 4, 5),
}

LAYER_CODES = np.array([0, 1], dtype=np.int8)
LAYER_NAMES = np.array(["400_600_hpa", "lowest_model_to_100_hpa"])
LAYER_TOP_HPA = np.array([400.0, 100.0], dtype=np.float64)
LAYER_BOTTOM_HPA = np.array([600.0, np.nan], dtype=np.float64)

G = rce.G


@dataclass
class TimeseriesConfig:
    """Configuration for the paired CTL/RCE_RH time-series workflow."""

    ctl_root: Path
    rh_root: Path
    cache_dir: Path
    output_path: Path
    start_day: int = 1
    end_day: int = 101
    samples_per_day: int = 4
    time_tolerance: float = 1.0e-6
    overwrite: bool = False
    inspect_only: bool = False
    verbose: bool = True

    def validate(self) -> None:
        if self.end_day <= self.start_day:
            raise ValueError("end_day must be greater than start_day")
        if int(self.start_day) != self.start_day or int(self.end_day) != self.end_day:
            raise ValueError("start_day and end_day must be integers")
        if self.samples_per_day <= 0:
            raise ValueError("samples_per_day must be positive")
        if self.time_tolerance <= 0:
            raise ValueError("time_tolerance must be positive")


def _time_key(value: float, tolerance: float) -> int:
    return int(np.rint(float(value) / tolerance))


def _experiment_slug(experiment_id: str) -> str:
    return experiment_id.lower()


def daily_cache_path(
    cache_dir: Path | str, experiment_id: str, day: int
) -> Path:
    return (
        Path(cache_dir)
        / "daily"
        / _experiment_slug(experiment_id)
        / f"rce_pclass_timeseries_day_{int(day):04d}.nc"
    )


def pressure_cell_interfaces(pressure_pa: np.ndarray) -> np.ndarray:
    """Return pressure interfaces for mass-level pressure columns.

    Pressure must decrease with increasing vertical index.  Invalid or
    nonmonotonic columns receive all-NaN interfaces.
    """

    pressure = np.asarray(pressure_pa, dtype=np.float64)
    if pressure.ndim != 3 or pressure.shape[0] < 2:
        raise ValueError("pressure_pa must have shape (z, y, x) with at least 2 levels")

    interfaces = np.empty((pressure.shape[0] + 1, *pressure.shape[1:]), dtype=np.float64)
    interfaces[1:-1] = 0.5 * (pressure[:-1] + pressure[1:])
    interfaces[0] = pressure[0] + 0.5 * (pressure[0] - pressure[1])
    interfaces[-1] = pressure[-1] - 0.5 * (pressure[-2] - pressure[-1])

    valid_columns = np.all(np.isfinite(pressure), axis=0) & np.all(
        np.diff(pressure, axis=0) < 0.0, axis=0
    )
    interfaces[:, ~valid_columns] = np.nan
    return interfaces


def pressure_layer_thickness(
    pressure_pa: np.ndarray,
    top_hpa: float,
    bottom_hpa: float | None,
) -> np.ndarray:
    """Pressure thickness of each mass layer overlapping requested bounds.

    ``bottom_hpa=None`` means the lower native model interface.  The requested
    top and finite bottom boundaries are clipped exactly.
    """

    if not np.isfinite(top_hpa) or top_hpa < 0:
        raise ValueError("top_hpa must be finite and nonnegative")
    if bottom_hpa is not None:
        if not np.isfinite(bottom_hpa) or bottom_hpa <= top_hpa:
            raise ValueError("bottom_hpa must be finite and greater than top_hpa")

    interfaces = pressure_cell_interfaces(pressure_pa)
    cell_bottom = interfaces[:-1]
    cell_top = interfaces[1:]
    requested_top = float(top_hpa) * 100.0
    if bottom_hpa is None:
        requested_bottom = interfaces[0][None, ...]
    else:
        requested_bottom = float(bottom_hpa) * 100.0

    overlap_bottom = np.minimum(cell_bottom, requested_bottom)
    overlap_top = np.maximum(cell_top, requested_top)
    thickness = np.maximum(0.0, overlap_bottom - overlap_top)
    invalid = ~np.isfinite(cell_bottom) | ~np.isfinite(cell_top)
    thickness[invalid] = np.nan
    return thickness


def layer_pressure_thicknesses(pressure_pa: np.ndarray) -> np.ndarray:
    """Return pressure weights for the two configured diagnostic layers."""

    return np.stack(
        [
            pressure_layer_thickness(pressure_pa, top_hpa=400.0, bottom_hpa=600.0),
            pressure_layer_thickness(pressure_pa, top_hpa=100.0, bottom_hpa=None),
        ],
        axis=0,
    )


def _integrate_tendency(
    values: np.ndarray, pressure_thickness_pa: np.ndarray
) -> np.ndarray:
    """Mass-integrate a mixing-ratio tendency through one pressure layer."""

    field = np.asarray(values, dtype=np.float64)
    thickness = np.asarray(pressure_thickness_pa, dtype=np.float64)
    if field.shape != thickness.shape:
        raise ValueError("values and pressure_thickness_pa must have identical shapes")

    contributing = np.isfinite(thickness) & (thickness > 0.0)
    complete = np.all(~contributing | np.isfinite(field), axis=0)
    layer_mass = np.sum(np.where(contributing, thickness, 0.0), axis=0)
    result = np.sum(
        np.where(contributing & np.isfinite(field), field * thickness / G, 0.0),
        axis=0,
    )
    result[(layer_mass <= 0.0) | ~complete] = np.nan
    return result


def compute_column_relative_humidity(
    vapor_mixing_ratio: np.ndarray,
    saturation_mixing_ratio: np.ndarray,
    pressure_thickness_pa: np.ndarray,
) -> np.ndarray:
    """Return integrated q/qsat for one layer in every horizontal column."""

    qv = np.asarray(vapor_mixing_ratio, dtype=np.float64)
    qsat = np.asarray(saturation_mixing_ratio, dtype=np.float64)
    thickness = np.asarray(pressure_thickness_pa, dtype=np.float64)
    if qv.shape != qsat.shape or qv.shape != thickness.shape:
        raise ValueError("qv, qsat, and pressure thickness must have identical shapes")

    contributing = np.isfinite(thickness) & (thickness > 0.0)
    complete = np.all(
        ~contributing | (np.isfinite(qv) & np.isfinite(qsat)), axis=0
    )
    numerator = np.sum(
        np.where(contributing & np.isfinite(qv), qv * thickness, 0.0), axis=0
    )
    denominator = np.sum(
        np.where(contributing & np.isfinite(qsat), qsat * thickness, 0.0), axis=0
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        result = numerator / denominator
    result[(denominator <= 0.0) | ~complete] = np.nan
    return result


def vertical_advective_moistening(
    vapor_mixing_ratio: np.ndarray,
    vertical_velocity: np.ndarray,
    height_m: np.ndarray,
) -> np.ndarray:
    """Return signed vapor moistening ``-w dqv/dz`` in s-1."""

    qv = np.asarray(vapor_mixing_ratio, dtype=np.float64)
    w = np.asarray(vertical_velocity, dtype=np.float64)
    height = np.asarray(height_m, dtype=np.float64).squeeze()
    if qv.shape != w.shape or qv.ndim != 3:
        raise ValueError("qv and vertical_velocity must match with shape (z, y, x)")
    if height.ndim != 1 or height.size != qv.shape[0]:
        raise ValueError("height_m must be one-dimensional and match qv levels")
    if not np.all(np.isfinite(height)) or not np.all(np.diff(height) > 0.0):
        raise ValueError("height_m must be finite and strictly increasing")

    edge_order = 2 if height.size >= 3 else 1
    dq_dz = np.gradient(qv, height, axis=0, edge_order=edge_order)
    return -w * dq_dz


def compute_mass_flux_columns(
    vertical_velocity: np.ndarray,
    pressure_thickness_pa: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return positive updraft and downdraft column mass-flux magnitudes."""

    w = np.asarray(vertical_velocity, dtype=np.float64)
    thickness = np.asarray(pressure_thickness_pa, dtype=np.float64)
    if w.shape != thickness.shape:
        raise ValueError("vertical_velocity and pressure thickness must match")
    updraft = _integrate_tendency(np.maximum(w, 0.0), thickness)
    downdraft = _integrate_tendency(np.maximum(-w, 0.0), thickness)
    return updraft, downdraft


def pclass_masks(pclass: np.ndarray) -> np.ndarray:
    """Return masks for DeepC, Strat, Anvil, and their DSA union."""

    classes = np.asarray(pclass)
    if classes.ndim != 2:
        raise ValueError("pclass must have shape (y, x)")
    return np.stack(
        [np.isin(classes, PCLASS_MEMBERS[int(code)]) for code in PCLASS_CODES],
        axis=0,
    )


def conditional_mean(
    values: np.ndarray, pclass: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Average one scalar or layer stack over each configured p-class mask.

    A two-dimensional input returns ``(pclass,)``.  A ``(layer, y, x)`` input
    returns ``(pclass, layer)``.
    """

    field = np.asarray(values, dtype=np.float64)
    classes = np.asarray(pclass)
    if field.ndim not in (2, 3) or field.shape[-2:] != classes.shape:
        raise ValueError("values must have shape (y,x) or (layer,y,x) matching pclass")

    masks = pclass_masks(classes)
    if field.ndim == 2:
        means = np.full(PCLASS_CODES.size, np.nan, dtype=np.float64)
        counts = np.zeros(PCLASS_CODES.size, dtype=np.int32)
        for index, mask in enumerate(masks):
            valid = mask & np.isfinite(field)
            counts[index] = np.count_nonzero(valid)
            if counts[index] > 0:
                means[index] = np.mean(field[valid], dtype=np.float64)
        return means, counts

    means = np.full((PCLASS_CODES.size, field.shape[0]), np.nan, dtype=np.float64)
    counts = np.zeros((PCLASS_CODES.size, field.shape[0]), dtype=np.int32)
    for index, mask in enumerate(masks):
        valid = mask[None, ...] & np.isfinite(field)
        counts[index] = np.count_nonzero(valid, axis=(1, 2))
        sums = np.sum(np.where(valid, field, 0.0), axis=(1, 2), dtype=np.float64)
        np.divide(sums, counts[index], out=means[index], where=counts[index] > 0)
    return means, counts


def precipitation_efficiency_from_columns(
    updraft_mass_flux: np.ndarray,
    downdraft_mass_flux: np.ndarray,
    pclass: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Average mass fluxes by class, then return raw ``1 - Md/Mu``.

    The ordering is intentional and reproduces ``precip_eff_rce.ipynb``: PE is
    calculated from class-mean updraft and downdraft flux, not independently in
    every column before horizontal averaging.
    """

    updraft, updraft_count = conditional_mean(updraft_mass_flux, pclass)
    downdraft, downdraft_count = conditional_mean(downdraft_mass_flux, pclass)
    count = np.minimum(updraft_count, downdraft_count)
    pe = np.full(PCLASS_CODES.size, np.nan, dtype=np.float64)
    valid = (count > 0) & np.isfinite(updraft) & (updraft > 0.0)
    pe[valid] = 1.0 - downdraft[valid] / updraft[valid]
    return pe, updraft, downdraft, count


def compute_time_diagnostics(dataset: xr.Dataset) -> dict[str, np.ndarray]:
    """Compute all requested diagnostics from one single-time SAM dataset."""

    pressure = rce.pressure_from_dataset(dataset)
    temperature = rce._squeezed_array(dataset, "TABS", 3)
    qv = rce.mixing_ratio_from_dataset(dataset, "QV")
    qn = rce.mixing_ratio_from_dataset(dataset, "QN")
    qp = rce.mixing_ratio_from_dataset(dataset, "QP")
    w = rce.align_vertical_velocity(
        rce._squeezed_array(dataset, "W", 3), pressure.shape[0]
    )
    height = rce._squeezed_array(dataset, "z", 1)

    expected_shape = pressure.shape
    for name, values in {
        "TABS": temperature,
        "QV": qv,
        "QN": qn,
        "QP": qp,
        "W": w,
    }.items():
        if values.shape != expected_shape:
            raise rce.RCEProcessingError(
                f"{name} shape {values.shape} does not match pressure {expected_shape}"
            )

    pclass = rce.generate_pclass(temperature, qn, qp, pressure)
    layer_dp = layer_pressure_thicknesses(pressure)
    full_layer_dp = layer_dp[1]

    updraft_column, downdraft_column = compute_mass_flux_columns(w, full_layer_dp)
    (
        precipitation_efficiency,
        updraft,
        downdraft,
        mass_flux_count,
    ) = precipitation_efficiency_from_columns(
        updraft_column, downdraft_column, pclass
    )

    qsat = rv_saturation(temperature, pressure)
    column_rh = np.stack(
        [compute_column_relative_humidity(qv, qsat, layer) for layer in layer_dp],
        axis=0,
    )
    relative_humidity, rh_count = conditional_mean(column_rh, pclass)

    gridpoint_moistening = vertical_advective_moistening(qv, w, height)
    column_moistening = np.stack(
        [_integrate_tendency(gridpoint_moistening, layer) for layer in layer_dp],
        axis=0,
    )
    vertical_moistening, vertical_moistening_count = conditional_mean(
        column_moistening, pclass
    )

    masks = pclass_masks(pclass)
    gridcell_count = np.count_nonzero(masks, axis=(1, 2)).astype(np.int32)
    valid_primitive = np.isin(pclass, np.arange(6))
    domain_count = int(np.count_nonzero(valid_primitive))
    area_fraction = np.full(PCLASS_CODES.size, np.nan, dtype=np.float64)
    if domain_count > 0:
        area_fraction = gridcell_count.astype(np.float64) / domain_count

    return {
        "precipitation_efficiency": precipitation_efficiency,
        "updraft_mass_flux": updraft,
        "downdraft_mass_flux": downdraft,
        "relative_humidity": relative_humidity,
        "vertical_advective_moistening": vertical_moistening,
        "pclass_gridcell_count": gridcell_count,
        "pclass_area_fraction": area_fraction,
        "mass_flux_valid_column_count": mass_flux_count,
        "relative_humidity_valid_column_count": rh_count,
        "vertical_advective_moistening_valid_column_count": vertical_moistening_count,
        "valid_domain_gridcell_count": np.asarray(domain_count, dtype=np.int32),
    }


def _daily_dataset(
    experiment_id: str,
    records: Sequence[rce.FileRecord],
    diagnostics: Sequence[Mapping[str, np.ndarray]],
) -> xr.Dataset:
    if not records or len(records) != len(diagnostics):
        raise ValueError("records and diagnostics must be nonempty and have equal length")

    def stack(name: str) -> np.ndarray:
        return np.stack([np.asarray(item[name]) for item in diagnostics], axis=0)

    dataset = xr.Dataset(
        data_vars={
            "precipitation_efficiency": (
                ("time", "pclass"),
                stack("precipitation_efficiency"),
            ),
            "updraft_mass_flux": (("time", "pclass"), stack("updraft_mass_flux")),
            "downdraft_mass_flux": (
                ("time", "pclass"),
                stack("downdraft_mass_flux"),
            ),
            "relative_humidity": (
                ("time", "pclass", "layer"),
                stack("relative_humidity"),
            ),
            "vertical_advective_moistening": (
                ("time", "pclass", "layer"),
                stack("vertical_advective_moistening"),
            ),
            "pclass_gridcell_count": (
                ("time", "pclass"),
                stack("pclass_gridcell_count"),
            ),
            "pclass_area_fraction": (
                ("time", "pclass"),
                stack("pclass_area_fraction"),
            ),
            "mass_flux_valid_column_count": (
                ("time", "pclass"),
                stack("mass_flux_valid_column_count"),
            ),
            "relative_humidity_valid_column_count": (
                ("time", "pclass", "layer"),
                stack("relative_humidity_valid_column_count"),
            ),
            "vertical_advective_moistening_valid_column_count": (
                ("time", "pclass", "layer"),
                stack("vertical_advective_moistening_valid_column_count"),
            ),
            "valid_domain_gridcell_count": (
                ("time",),
                stack("valid_domain_gridcell_count"),
            ),
            "source_file": (
                ("time",),
                np.asarray([str(record.path) for record in records]),
            ),
        },
        coords={
            "time": np.asarray([record.time for record in records], dtype=np.float64),
            "pclass": PCLASS_CODES,
            "layer": LAYER_CODES,
            "pclass_name": (("pclass",), PCLASS_NAMES),
            "layer_name": (("layer",), LAYER_NAMES),
            "layer_top_pressure_hpa": (("layer",), LAYER_TOP_HPA),
            "layer_bottom_pressure_hpa": (("layer",), LAYER_BOTTOM_HPA),
        },
        attrs={
            "schema_version": SCHEMA_VERSION,
            "diagnostic_version": DIAGNOSTIC_VERSION,
            "classifier_version": rce.CLASSIFIER_VERSION,
            "experiment_id": experiment_id,
            "description": "SAM RCE time series conditioned on precipitation class",
            "pclass_membership": json.dumps(
                {str(key): list(value) for key, value in PCLASS_MEMBERS.items()},
                sort_keys=True,
            ),
            "horizontal_weighting": "equal-area mean of all finite masked grid cells",
            "pe_definition": "1 - mean(Md_column) / mean(Mu_column)",
            "mass_flux_vertical_bounds": "lowest native model layer to exactly 100 hPa",
            "relative_humidity_definition": "sum(qv*dp) / sum(qv_sat*dp)",
            "vertical_advective_moistening_definition": "sum((-W*dQV/dz)*dp/g)",
            "vertical_advective_moistening_sign": "positive moistening; signed values retained",
            "layer_0_definition": "exactly 400 to 600 hPa",
            "layer_1_definition": "complete lowest native model layer to exactly 100 hPa",
            "pe_clipping": "none",
            "relative_humidity_clipping": "none",
        },
    )

    dataset["time"].attrs.update(units="day", long_name="SAM model time")
    dataset["pclass"].attrs.update(
        long_name="precipitation class code; 6 is composite DSA"
    )
    dataset["layer_top_pressure_hpa"].attrs["units"] = "hPa"
    dataset["layer_bottom_pressure_hpa"].attrs.update(
        units="hPa",
        note="NaN denotes the lower native model-layer interface",
    )
    dataset["precipitation_efficiency"].attrs.update(
        long_name="mass-flux precipitation efficiency", units="1"
    )
    for name, direction in (
        ("updraft_mass_flux", "upward"),
        ("downdraft_mass_flux", "downward magnitude"),
    ):
        dataset[name].attrs.update(
            long_name=f"class-mean vertically integrated {direction} mass flux",
            units="kg m-1 s-1",
        )
    dataset["relative_humidity"].attrs.update(
        long_name="class-mean integrated layer relative humidity", units="1"
    )
    dataset["vertical_advective_moistening"].attrs.update(
        long_name="class-mean layer-integrated vertical advective vapor moistening",
        units="kg m-2 s-1",
    )
    dataset["pclass_area_fraction"].attrs.update(
        long_name="fraction of valid domain grid cells in pclass", units="1"
    )
    return dataset


def _encoding(dataset: xr.Dataset) -> dict[str, dict]:
    encoding: dict[str, dict] = {}
    count_names = {
        "pclass_gridcell_count",
        "mass_flux_valid_column_count",
        "relative_humidity_valid_column_count",
        "vertical_advective_moistening_valid_column_count",
        "valid_domain_gridcell_count",
    }
    float_names = {
        "precipitation_efficiency",
        "updraft_mass_flux",
        "downdraft_mass_flux",
        "relative_humidity",
        "vertical_advective_moistening",
        "pclass_area_fraction",
    }
    for name in dataset.data_vars:
        if name in float_names:
            encoding[name] = {
                "zlib": True,
                "complevel": 2,
                "shuffle": True,
                "dtype": "float32",
                "_FillValue": np.float32(np.nan),
            }
        elif name in count_names:
            encoding[name] = {
                "zlib": True,
                "complevel": 2,
                "shuffle": True,
                "dtype": "int32",
            }
    return encoding


def _discover_experiment_records(
    root: Path,
    config: TimeseriesConfig,
) -> tuple[list[rce.FileRecord], list[rce.FileRecord]]:
    # SAM encodes model time in every single-time 3-D filename. Build the index
    # from those hints so whole-period discovery does not open hundreds of files
    # merely to read one scalar. The internal time is checked when a selected
    # file is opened for inspection or processing.
    records: list[rce.FileRecord] = []
    excluded: list[rce.FileRecord] = []
    directory = rce._resolve_3d_directory(root)
    paths = sorted(directory.glob("*.nc"))
    if not paths:
        raise FileNotFoundError(f"No NetCDF files found in {directory}")
    for path in paths:
        hint = rce._filename_time_hint(path)
        if hint is None:
            raise rce.RCEProcessingError(
                f"Could not infer SAM model time from 3-D filename {path.name}"
            )
        if not config.start_day <= hint < config.end_day:
            continue
        phase = (hint - math.floor(hint)) * config.samples_per_day
        if np.isclose(phase, np.rint(phase), atol=config.time_tolerance, rtol=0.0):
            records.append(rce.FileRecord(hint, path))
        else:
            excluded.append(rce.FileRecord(hint, path))
    records.sort(key=lambda record: record.time)
    excluded.sort(key=lambda record: record.time)

    keys = [_time_key(record.time, config.time_tolerance) for record in records]
    if len(keys) != len(set(keys)):
        raise rce.RCEProcessingError(f"Duplicate regular 3-D times below {root}")
    return records, excluded


def _validate_record_time(
    dataset: xr.Dataset, record: rce.FileRecord, tolerance: float
) -> None:
    if "time" not in dataset:
        raise rce.RCEProcessingError(f"Required time coordinate is absent in {record.path}")
    value = np.asarray(dataset["time"].values).squeeze()
    if np.ndim(value) != 0 or not np.isfinite(value):
        raise rce.RCEProcessingError(f"Expected one finite scalar time in {record.path}")
    if not np.isclose(float(value), record.time, atol=tolerance, rtol=0.0):
        raise rce.RCEProcessingError(
            f"Filename time {record.time} does not match internal time {float(value)} "
            f"in {record.path}"
        )


def _validate_regular_window(
    records: Sequence[rce.FileRecord], config: TimeseriesConfig, experiment_id: str
) -> None:
    phases = np.arange(config.samples_per_day, dtype=np.float64) / config.samples_per_day
    errors: list[str] = []
    for day in range(config.start_day, config.end_day):
        actual = np.asarray(
            [record.time for record in records if day <= record.time < day + 1],
            dtype=np.float64,
        )
        expected = day + phases
        complete = actual.size == expected.size and np.allclose(
            actual, expected, atol=config.time_tolerance, rtol=0.0
        )
        terminal = (
            day == config.end_day - 1
            and actual.size == 1
            and np.isclose(actual[0], day, atol=config.time_tolerance, rtol=0.0)
        )
        if not complete and not terminal:
            errors.append(f"day {day}: found {actual.tolist()}, expected {expected.tolist()}")
    if errors:
        raise rce.RCEProcessingError(
            f"Incomplete regular sampling for {experiment_id}:\n" + "\n".join(errors)
        )


def discover_paired_records(
    config: TimeseriesConfig,
) -> tuple[dict[str, list[rce.FileRecord]], dict[str, list[rce.FileRecord]]]:
    """Discover regular records and require exact CTL/RCE_RH time alignment."""

    selected: dict[str, list[rce.FileRecord]] = {}
    excluded: dict[str, list[rce.FileRecord]] = {}
    for experiment_id, root in (
        ("RCE_CTL", config.ctl_root),
        ("RCE_RH", config.rh_root),
    ):
        selected[experiment_id], excluded[experiment_id] = _discover_experiment_records(
            root, config
        )
        _validate_regular_window(selected[experiment_id], config, experiment_id)

    ctl_keys = {
        _time_key(record.time, config.time_tolerance): record
        for record in selected["RCE_CTL"]
    }
    rh_keys = {
        _time_key(record.time, config.time_tolerance): record
        for record in selected["RCE_RH"]
    }
    if ctl_keys.keys() != rh_keys.keys():
        ctl_only = sorted(record.time for key, record in ctl_keys.items() if key not in rh_keys)
        rh_only = sorted(record.time for key, record in rh_keys.items() if key not in ctl_keys)
        raise rce.RCEProcessingError(
            f"Experiment times do not match; CTL-only={ctl_only}, RCE_RH-only={rh_only}"
        )
    if not ctl_keys:
        raise rce.RCEProcessingError("No matched regular records were selected")
    return selected, excluded


def _validate_daily_cache(
    path: Path,
    experiment_id: str,
    records: Sequence[rce.FileRecord],
    tolerance: float,
) -> None:
    try:
        with xr.open_dataset(path, decode_times=False) as dataset:
            if dataset.attrs.get("schema_version") != SCHEMA_VERSION:
                raise rce.CacheValidationError(f"Unsupported schema in {path}")
            if dataset.attrs.get("diagnostic_version") != DIAGNOSTIC_VERSION:
                raise rce.CacheValidationError(f"Unsupported diagnostics in {path}")
            if dataset.attrs.get("classifier_version") != rce.CLASSIFIER_VERSION:
                raise rce.CacheValidationError(f"Unsupported classifier in {path}")
            if dataset.attrs.get("experiment_id") != experiment_id:
                raise rce.CacheValidationError(f"Experiment mismatch in {path}")
            expected_times = np.asarray([record.time for record in records])
            actual_times = np.asarray(dataset["time"].values, dtype=float)
            if actual_times.shape != expected_times.shape or not np.allclose(
                actual_times, expected_times, atol=tolerance, rtol=0.0
            ):
                raise rce.CacheValidationError(f"Time mismatch in {path}")
            expected_files = [record.path.name for record in records]
            actual_files = [Path(str(value)).name for value in dataset["source_file"].values]
            if actual_files != expected_files:
                raise rce.CacheValidationError(f"Source-file mismatch in {path}")
    except rce.CacheValidationError:
        raise
    except Exception as exc:
        raise rce.CacheValidationError(f"Could not validate {path}: {exc}") from exc


def _group_by_day(records: Sequence[rce.FileRecord]) -> dict[int, list[rce.FileRecord]]:
    grouped: dict[int, list[rce.FileRecord]] = {}
    for record in records:
        grouped.setdefault(math.floor(record.time), []).append(record)
    return {day: sorted(values, key=lambda record: record.time) for day, values in grouped.items()}


def preprocess_daily_caches(
    config: TimeseriesConfig,
    selected: Mapping[str, Sequence[rce.FileRecord]] | None = None,
) -> dict[str, list[Path]]:
    """Build or safely reuse all requested daily time-series caches."""

    config.validate()
    if selected is None:
        selected, _ = discover_paired_records(config)
    written: dict[str, list[Path]] = {key: [] for key in EXPERIMENT_IDS}

    for experiment_id in EXPERIMENT_IDS:
        for day, records in _group_by_day(selected[experiment_id]).items():
            output_path = daily_cache_path(config.cache_dir, experiment_id, day)
            if output_path.exists() and not config.overwrite:
                try:
                    _validate_daily_cache(
                        output_path,
                        experiment_id,
                        records,
                        config.time_tolerance,
                    )
                except rce.CacheValidationError as exc:
                    raise rce.CacheValidationError(
                        f"{exc}. Re-run with --overwrite to replace this cache."
                    ) from exc
                if config.verbose:
                    print(f"Reusing {output_path}", flush=True)
                written[experiment_id].append(output_path)
                continue

            if config.verbose:
                print(
                    f"Processing {experiment_id} day {day} ({len(records)} times)",
                    flush=True,
                )
            diagnostics: list[dict[str, np.ndarray]] = []
            for record in records:
                if config.verbose:
                    print(f"  {record.time:8.2f}  {record.path.name}", flush=True)
                with xr.open_dataset(record.path, decode_times=False) as dataset:
                    _validate_record_time(dataset, record, config.time_tolerance)
                    diagnostics.append(compute_time_diagnostics(dataset))
            daily = _daily_dataset(experiment_id, records, diagnostics)
            rce.atomic_to_netcdf(daily, output_path, _encoding(daily))
            written[experiment_id].append(output_path)
    return written


def aggregate_daily_caches(
    paths: Mapping[str, Sequence[Path | str]],
) -> xr.Dataset:
    """Concatenate daily caches into an experiment-aligned time series."""

    experiment_datasets: list[xr.Dataset] = []
    reference_times: np.ndarray | None = None
    for experiment_id in EXPERIMENT_IDS:
        experiment_paths = [Path(path) for path in paths.get(experiment_id, [])]
        if not experiment_paths:
            raise rce.RCEProcessingError(f"No daily caches supplied for {experiment_id}")
        loaded: list[xr.Dataset] = []
        for path in sorted(experiment_paths):
            with xr.open_dataset(path, decode_times=False) as dataset:
                if dataset.attrs.get("schema_version") != SCHEMA_VERSION:
                    raise rce.CacheValidationError(f"Unsupported schema in {path}")
                if dataset.attrs.get("experiment_id") != experiment_id:
                    raise rce.CacheValidationError(f"Experiment mismatch in {path}")
                loaded.append(dataset.load())
        combined = xr.concat(
            loaded,
            dim="time",
            data_vars="minimal",
            coords="minimal",
            compat="equals",
            combine_attrs="override",
        ).sortby("time")
        times = np.asarray(combined["time"].values, dtype=float)
        if np.unique(times).size != times.size:
            raise rce.CacheValidationError(f"Duplicate times in {experiment_id} caches")
        if reference_times is None:
            reference_times = times
        elif times.shape != reference_times.shape or not np.allclose(
            times, reference_times, atol=1.0e-6, rtol=0.0
        ):
            raise rce.CacheValidationError("Daily CTL/RCE_RH cache times do not align")
        combined = combined.expand_dims(experiment=[experiment_id])
        experiment_datasets.append(combined)

    output = xr.concat(
        experiment_datasets,
        dim="experiment",
        data_vars="all",
        coords="minimal",
        compat="equals",
        combine_attrs="override",
    )
    output.attrs.update(
        schema_version=SCHEMA_VERSION,
        diagnostic_version=DIAGNOSTIC_VERSION,
        classifier_version=rce.CLASSIFIER_VERSION,
        description="Paired SAM RCE pclass-conditioned diagnostic time series",
        experiments=json.dumps(list(EXPERIMENT_IDS)),
        time_alignment="exact matched regular quarter-day SAM records",
    )
    output.attrs.pop("experiment_id", None)
    output["experiment"].attrs["long_name"] = "SAM experiment"
    return output


def write_aggregate(dataset: xr.Dataset, path: Path | str) -> Path:
    """Atomically write the final compressed time-series dataset."""

    return rce.atomic_to_netcdf(dataset, Path(path), _encoding(dataset))


def _validate_aggregate_output(
    path: Path,
    selected: Mapping[str, Sequence[rce.FileRecord]],
    tolerance: float,
) -> None:
    """Require an existing aggregate to match the current paired inputs."""

    try:
        with xr.open_dataset(path, decode_times=False) as dataset:
            if dataset.attrs.get("schema_version") != SCHEMA_VERSION:
                raise rce.CacheValidationError(f"Unsupported schema in {path}")
            if dataset.attrs.get("diagnostic_version") != DIAGNOSTIC_VERSION:
                raise rce.CacheValidationError(f"Unsupported diagnostics in {path}")
            experiments = [str(value) for value in dataset["experiment"].values]
            if experiments != list(EXPERIMENT_IDS):
                raise rce.CacheValidationError(f"Experiment mismatch in {path}")
            expected_times = np.asarray(
                [record.time for record in selected["RCE_CTL"]], dtype=float
            )
            actual_times = np.asarray(dataset["time"].values, dtype=float)
            if actual_times.shape != expected_times.shape or not np.allclose(
                actual_times, expected_times, atol=tolerance, rtol=0.0
            ):
                raise rce.CacheValidationError(f"Time mismatch in {path}")
            for experiment_id in EXPERIMENT_IDS:
                expected_files = [record.path.name for record in selected[experiment_id]]
                actual_files = [
                    Path(str(value)).name
                    for value in dataset["source_file"]
                    .sel(experiment=experiment_id)
                    .values
                ]
                if actual_files != expected_files:
                    raise rce.CacheValidationError(
                        f"Source-file mismatch for {experiment_id} in {path}"
                    )
    except rce.CacheValidationError:
        raise
    except Exception as exc:
        raise rce.CacheValidationError(f"Could not validate {path}: {exc}") from exc


def inspect_inputs(
    config: TimeseriesConfig,
    selected: Mapping[str, Sequence[rce.FileRecord]],
    excluded: Mapping[str, Sequence[rce.FileRecord]],
) -> dict[str, object]:
    inventory: dict[str, object] = {
        "window": [config.start_day, config.end_day],
        "window_convention": "half-open",
        "matched_times": [record.time for record in selected["RCE_CTL"]],
        "experiments": {},
        "layers": {
            "400_600_hpa": [400.0, 600.0],
            "lowest_model_to_100_hpa": ["lowest native model layer", 100.0],
        },
        "time_index": "SAM filename time hints; internal time validated when opened",
    }
    for experiment_id, root in (
        ("RCE_CTL", config.ctl_root),
        ("RCE_RH", config.rh_root),
    ):
        record = selected[experiment_id][0]
        with xr.open_dataset(record.path, decode_times=False) as dataset:
            _validate_record_time(dataset, record, config.time_tolerance)
            variables = {}
            for name in ("W", "QV", "TABS", "QN", "QP", "PP", "p", "z"):
                variable = dataset[name]
                variables[name] = {
                    "dims": list(variable.dims),
                    "shape": list(variable.shape),
                    "units": str(variable.attrs.get("units", "")),
                }
        inventory["experiments"][experiment_id] = {
            "root": str(root),
            "selected_count": len(selected[experiment_id]),
            "excluded_irregular_count": len(excluded[experiment_id]),
            "first_source": str(record.path),
            "variables": variables,
        }
    return inventory


def run_workflow(config: TimeseriesConfig) -> Path | None:
    """Discover inputs, build daily caches, and write the aggregate dataset."""

    config.validate()
    selected, excluded = discover_paired_records(config)
    if config.inspect_only:
        print(json.dumps(inspect_inputs(config, selected, excluded), indent=2))
        return None
    daily_paths = preprocess_daily_caches(config, selected)
    if config.output_path.exists() and not config.overwrite:
        try:
            _validate_aggregate_output(
                config.output_path, selected, config.time_tolerance
            )
        except rce.CacheValidationError as exc:
            raise rce.CacheValidationError(
                f"{exc}. Re-run with --overwrite to replace this aggregate."
            ) from exc
        if config.verbose:
            print(f"Reusing {config.output_path}", flush=True)
        return config.output_path
    aggregate = aggregate_daily_caches(daily_paths)
    aggregate.attrs.update(
        ctl_root=str(config.ctl_root),
        rce_rh_root=str(config.rh_root),
        start_day_inclusive=int(config.start_day),
        end_day_exclusive=int(config.end_day),
    )
    return write_aggregate(aggregate, config.output_path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build matched CTL/RCE_RH SAM time series of precipitation efficiency, "
            "layer relative humidity, and vertical advective moistening by pclass."
        )
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--ctl-root", type=Path)
    parser.add_argument("--rh-root", type=Path)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--start-day", type=int, default=1)
    parser.add_argument("--end-day", type=int, default=101)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace existing daily caches and aggregate output",
    )
    parser.add_argument(
        "--inspect",
        action="store_true",
        help="validate and print the paired time/variable inventory without writing",
    )
    parser.add_argument("--quiet", action="store_true")
    return parser


def config_from_args(args: argparse.Namespace) -> TimeseriesConfig:
    data_root = Path(args.data_root)
    cache_dir = (
        Path(args.cache_dir)
        if args.cache_dir
        else data_root / "pickle_out" / "pclass_timeseries_rce"
    )
    output_path = (
        Path(args.output)
        if args.output
        else cache_dir / "rce_pclass_diagnostics_timeseries.nc"
    )
    return TimeseriesConfig(
        ctl_root=Path(args.ctl_root) if args.ctl_root else data_root / DEFAULT_CTL_NAME,
        rh_root=Path(args.rh_root) if args.rh_root else data_root / DEFAULT_RH_NAME,
        cache_dir=cache_dir,
        output_path=output_path,
        start_day=args.start_day,
        end_day=args.end_day,
        overwrite=args.overwrite,
        inspect_only=args.inspect,
        verbose=not args.quiet,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    config = config_from_args(args)
    try:
        output = run_workflow(config)
    except (OSError, ValueError, rce.RCEProcessingError) as exc:
        parser.error(str(exc))
    if output is not None and not args.quiet:
        print(f"Ready: {output}")
    return 0


__all__ = [
    "DIAGNOSTIC_VERSION",
    "EXPERIMENT_IDS",
    "LAYER_NAMES",
    "PCLASS_CODES",
    "PCLASS_NAMES",
    "SCHEMA_VERSION",
    "TimeseriesConfig",
    "aggregate_daily_caches",
    "compute_column_relative_humidity",
    "compute_mass_flux_columns",
    "compute_time_diagnostics",
    "conditional_mean",
    "daily_cache_path",
    "discover_paired_records",
    "layer_pressure_thicknesses",
    "main",
    "pclass_masks",
    "precipitation_efficiency_from_columns",
    "preprocess_daily_caches",
    "pressure_cell_interfaces",
    "pressure_layer_thickness",
    "run_workflow",
    "vertical_advective_moistening",
    "write_aggregate",
]


if __name__ == "__main__":
    raise SystemExit(main())
