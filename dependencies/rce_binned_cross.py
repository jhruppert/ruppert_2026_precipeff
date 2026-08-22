"""CRH-binned SAM RCE diagnostics with CTL LW-ACRE and LW-heating anomalies.

The raw-data path in this module is deliberately streaming: every model time is
read once, reduced to sums and counts, and written to a daily NetCDF file.  The
daily files can then be combined for arbitrary half-open day ranges without
re-reading the large three-dimensional model output.

The cloud classifier intentionally reproduces the implementation embedded in
``precip_eff_rce.ipynb``.  In particular, the legacy snow/graupel partition is
preserved so regenerated labels agree with existing ``pclass`` pickles.  The
classification inputs are never retained in the daily statistics.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import pickle
import re
import tempfile
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import xarray as xr

from dependencies.thermo_functions import rv_saturation


SCHEMA_VERSION = "3.1"
CLASSIFIER_VERSION = "sam_pclass_legacy_v1"

DEFAULT_DATA_ROOT = Path(
    "/ourdisk/hpc/radclouds/auto_archive_notyet/tape_2copies/"
    "jruppert/wing_rce"
)
DEFAULT_CTL_NAME = "RCE_768_305_snd_newoutput"

CRH_EDGES = np.round(np.arange(0.30, 1.00 + 0.01, 0.01), 2)
PRESSURE_HPA = np.arange(1000.0, 25.0, -25.0)
CLASS_CODES = np.arange(6, dtype=np.int8)
CLASS_NAMES = np.array(
    ["Non-cloud", "Deep", "Congestus", "Shallow", "Stratiform", "Anvil"]
)
DISPLAY_CLASS_CODES = (1, 4, 5)

G = 9.81
CP = 1004.0
RD = 287.05
EPSILON = 0.622
SECONDS_PER_DAY = 86400.0
PCLASS_FILL_VALUE = np.uint8(255)

# Exact-name allowlist only.  QRAD is intentionally absent because it is total
# radiative heating in the existing notebook.
LW_HEATING_ALLOWLIST = (
    "LQRAD",
    "QRADLW",
    "LWQRAD",
    "RADLW",
    "QRL",
    "LW_HEATING",
    "LWHEAT",
)
LW_ACRE_FLUX_VARIABLES = ("LWNS", "LWNT", "LWNSC", "LWNTC")


class RCEProcessingError(RuntimeError):
    """Raised when input data cannot satisfy the processing contract."""


class CacheValidationError(RCEProcessingError):
    """Raised when an existing cache cannot be safely reused."""


@dataclass(frozen=True)
class FileRecord:
    """A scalar SAM model time and its single-time 3-D file."""

    time: float
    path: Path


@dataclass
class PreprocessConfig:
    """Configuration for daily preprocessing."""

    ctl_root: Path
    cache_dir: Path
    pickle_dir: Path
    start_day: int = 1
    end_day: int = 21
    lw_var: str | None = None
    lw_units: str | None = None
    rain_var: str = "Prec"
    rain_units: str | None = None
    samples_per_day: int = 4
    time_tolerance: float = 1.0e-6
    overwrite: bool = False
    inspect_only: bool = False
    verbose: bool = True

    def validate(self) -> None:
        if self.end_day <= self.start_day:
            raise ValueError("end_day must be greater than start_day")
        if int(self.start_day) != self.start_day or int(self.end_day) != self.end_day:
            raise ValueError("daily caches require integer start_day and end_day")
        if self.samples_per_day <= 0:
            raise ValueError("samples_per_day must be positive")


def window_tag(start_day: int, end_day: int) -> str:
    """Return a filename-safe half-open day-window tag."""

    return f"days_{int(start_day):04d}_{int(end_day):04d}excl"


def daily_cache_path(cache_dir: Path | str, day: int) -> Path:
    return Path(cache_dir) / "daily" / f"rce_binned_day_{int(day):04d}.nc"


def pclass_cache_path(cache_dir: Path | str, day: int) -> Path:
    return Path(cache_dir) / "pclass" / f"rce_ctl_pclass_day_{int(day):04d}.nc"


def aggregate_output_path(
    output_dir: Path | str, start_day: int, end_day: int
) -> Path:
    return Path(output_dir) / f"rce_ctl_lw_anomaly_crh_{window_tag(start_day, end_day)}.nc"


def _time_key(value: float, tolerance: float) -> int:
    return int(np.rint(float(value) / tolerance))


def select_half_open_times(
    times: Sequence[float], start_day: float, end_day: float
) -> np.ndarray:
    """Select model times in ``[start_day, end_day)``."""

    values = np.asarray(times, dtype=float)
    return values[(values >= start_day) & (values < end_day)]


def assign_crh_bins(
    crh: np.ndarray, edges: np.ndarray = CRH_EDGES
) -> np.ndarray:
    """Return lower-inclusive, upper-exclusive bin indices, or -1 outside."""

    values = np.asarray(crh, dtype=float)
    indices = np.searchsorted(edges, values, side="right") - 1
    valid = (
        np.isfinite(values)
        & (values >= edges[0])
        & (values < edges[-1])
        & (indices >= 0)
        & (indices < len(edges) - 1)
    )
    return np.where(valid, indices, -1).astype(np.int16)


def binned_scalar_sums(
    values: np.ndarray, bin_index: np.ndarray, nbins: int
) -> tuple[np.ndarray, np.ndarray]:
    """Accumulate scalar sums and finite-value counts by bin."""

    value_flat = np.asarray(values, dtype=float).ravel()
    bin_flat = np.asarray(bin_index).ravel()
    valid = (bin_flat >= 0) & np.isfinite(value_flat)
    sums = np.bincount(
        bin_flat[valid], weights=value_flat[valid], minlength=nbins
    ).astype(np.float64)
    counts = np.bincount(bin_flat[valid], minlength=nbins).astype(np.int64)
    return sums[:nbins], counts[:nbins]


def binned_profile_sums(
    values: np.ndarray, bin_index: np.ndarray, nbins: int
) -> tuple[np.ndarray, np.ndarray]:
    """Accumulate ``(level, y, x)`` profile sums and counts by bin."""

    profile = np.asarray(values, dtype=float)
    if profile.ndim < 2:
        raise ValueError("profile values must include a vertical dimension")
    flat = profile.reshape(profile.shape[0], -1)
    bin_flat = np.asarray(bin_index).ravel()
    if flat.shape[1] != bin_flat.size:
        raise ValueError("profile columns and bin_index do not have the same size")

    sums = np.zeros((flat.shape[0], nbins), dtype=np.float64)
    counts = np.zeros((flat.shape[0], nbins), dtype=np.int64)
    in_bin = bin_flat >= 0
    for ilev in range(flat.shape[0]):
        valid = in_bin & np.isfinite(flat[ilev])
        sums[ilev] = np.bincount(
            bin_flat[valid], weights=flat[ilev, valid], minlength=nbins
        )[:nbins]
        counts[ilev] = np.bincount(bin_flat[valid], minlength=nbins)[:nbins]
    return sums, counts


def binned_class_counts(
    pclass: np.ndarray,
    bin_index: np.ndarray,
    nbins: int,
    nclasses: int = 6,
) -> np.ndarray:
    """Count primitive precipitation classes by CRH bin."""

    classes = np.asarray(pclass).ravel()
    bins = np.asarray(bin_index).ravel()
    valid = (
        (bins >= 0)
        & np.isfinite(classes)
        & (classes >= 0)
        & (classes < nclasses)
        & (classes == np.floor(classes))
    )
    combined = classes[valid].astype(np.int64) * nbins + bins[valid]
    counts = np.bincount(combined, minlength=nclasses * nbins)
    return counts.reshape(nclasses, nbins).astype(np.int64)


def safe_mean(sums: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """Divide sums by counts, returning NaN where no samples exist."""

    result = np.full(np.broadcast_shapes(np.shape(sums), np.shape(counts)), np.nan)
    np.divide(sums, counts, out=result, where=np.asarray(counts) > 0)
    return result


def mixing_ratio_to_virtual_temperature(
    temperature_k: np.ndarray, vapor_mixing_ratio: np.ndarray
) -> np.ndarray:
    """Virtual temperature for water-vapor mixing ratio (kg/kg dry air)."""

    temperature = np.asarray(temperature_k, dtype=float)
    mixing_ratio = np.asarray(vapor_mixing_ratio, dtype=float)
    return temperature * (1.0 + mixing_ratio / EPSILON) / (1.0 + mixing_ratio)


def density_from_state(
    pressure_pa: np.ndarray,
    temperature_k: np.ndarray,
    vapor_mixing_ratio: np.ndarray,
) -> np.ndarray:
    """Moist-air density from pressure and virtual temperature."""

    virtual_temperature = mixing_ratio_to_virtual_temperature(
        temperature_k, vapor_mixing_ratio
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        density = np.asarray(pressure_pa, dtype=float) / (RD * virtual_temperature)
    return np.where((density > 0) & np.isfinite(density), density, np.nan)


def align_vertical_velocity(w: np.ndarray, target_nz: int) -> np.ndarray:
    """Return W on mass levels, destaggering a single extra interface if needed."""

    values = np.asarray(w, dtype=float)
    if values.shape[0] == target_nz:
        return values
    if values.shape[0] == target_nz + 1:
        return 0.5 * (values[:-1] + values[1:])
    raise RCEProcessingError(
        f"W has {values.shape[0]} vertical levels; expected {target_nz} "
        f"or {target_nz + 1} staggered levels"
    )


def interpolate_fields_to_pressure(
    fields: Mapping[str, np.ndarray],
    pressure_pa: np.ndarray,
    target_pressure_hpa: np.ndarray = PRESSURE_HPA,
    chunk_columns: int = 8192,
) -> dict[str, np.ndarray]:
    """Linearly interpolate column fields to fixed pressure without extrapolation.

    A vectorized binary search is used for each column, avoiding Python loops over
    the 256 x 256 horizontal grid while allowing perturbation pressure to vary by
    column.
    """

    pressure = np.asarray(pressure_pa, dtype=float)
    if pressure.ndim != 3:
        raise ValueError("pressure_pa must have shape (z, y, x)")
    nz, ny, nx = pressure.shape
    flat_pressure = pressure.reshape(nz, -1)

    prepared: dict[str, np.ndarray] = {}
    for name, field in fields.items():
        values = np.asarray(field, dtype=float)
        if values.shape != pressure.shape:
            raise ValueError(
                f"{name} shape {values.shape} does not match pressure {pressure.shape}"
            )
        prepared[name] = values.reshape(nz, -1)

    median_gradient = np.nanmedian(np.diff(flat_pressure, axis=0))
    if median_gradient > 0:
        flat_pressure = flat_pressure[::-1]
        prepared = {name: values[::-1] for name, values in prepared.items()}

    targets = np.asarray(target_pressure_hpa, dtype=float) * 100.0
    outputs = {
        name: np.full((targets.size, flat_pressure.shape[1]), np.nan, dtype=np.float64)
        for name in prepared
    }

    for start in range(0, flat_pressure.shape[1], chunk_columns):
        stop = min(start + chunk_columns, flat_pressure.shape[1])
        p = flat_pressure[:, start:stop]
        ncol = p.shape[1]
        monotonic = np.all(np.diff(p, axis=0) < 0, axis=0)
        finite_pressure = np.all(np.isfinite(p), axis=0)

        target_matrix = np.broadcast_to(targets[:, None], (targets.size, ncol))
        valid = (
            monotonic[None, :]
            & finite_pressure[None, :]
            & (target_matrix <= p[0][None, :])
            & (target_matrix >= p[-1][None, :])
        )

        lower = np.zeros((targets.size, ncol), dtype=np.int16)
        upper = np.full((targets.size, ncol), nz - 1, dtype=np.int16)
        for _ in range(int(math.ceil(math.log2(max(nz, 2)))) + 1):
            unresolved = upper - lower > 1
            if not np.any(unresolved):
                break
            middle = ((lower.astype(np.int32) + upper.astype(np.int32)) // 2).astype(
                np.int16
            )
            p_middle = np.take_along_axis(p, middle, axis=0)
            move_lower = (p_middle >= target_matrix) & unresolved
            lower = np.where(move_lower, middle, lower)
            upper = np.where((~move_lower) & unresolved, middle, upper)

        p0 = np.take_along_axis(p, lower, axis=0)
        p1 = np.take_along_axis(p, upper, axis=0)
        denominator = p1 - p0
        with np.errstate(divide="ignore", invalid="ignore"):
            fraction = (target_matrix - p0) / denominator
        valid &= np.isfinite(fraction) & (denominator != 0)

        for name, values in prepared.items():
            value_chunk = values[:, start:stop]
            v0 = np.take_along_axis(value_chunk, lower, axis=0)
            v1 = np.take_along_axis(value_chunk, upper, axis=0)
            interpolated = v0 + fraction * (v1 - v0)
            interpolated[~(valid & np.isfinite(v0) & np.isfinite(v1))] = np.nan
            outputs[name][:, start:stop] = interpolated

    return {
        name: values.reshape(targets.size, ny, nx) for name, values in outputs.items()
    }


def interpolate_profile_to_pressure(
    values: np.ndarray,
    pressure_hpa: np.ndarray,
    target_pressure_hpa: np.ndarray = PRESSURE_HPA,
) -> np.ndarray:
    """Interpolate a single profile in pressure without extrapolation."""

    profile = np.asarray(values, dtype=float).squeeze()
    pressure = np.asarray(pressure_hpa, dtype=float).squeeze()
    if profile.ndim != 1 or pressure.ndim != 1 or profile.size != pressure.size:
        raise ValueError("values and pressure_hpa must be matching one-dimensional arrays")
    finite = np.isfinite(profile) & np.isfinite(pressure)
    if np.count_nonzero(finite) < 2:
        return np.full(np.asarray(target_pressure_hpa).shape, np.nan)
    order = np.argsort(pressure[finite])
    source_pressure = pressure[finite][order]
    source_values = profile[finite][order]
    return np.interp(
        np.asarray(target_pressure_hpa, dtype=float),
        source_pressure,
        source_values,
        left=np.nan,
        right=np.nan,
    )


def compute_crh(
    vapor_mixing_ratio: np.ndarray,
    temperature_k: np.ndarray,
    pressure_pa: np.ndarray,
) -> np.ndarray:
    """Reproduce the full-column CRH calculation in ``precip_eff_rce.ipynb``."""

    qv = np.asarray(vapor_mixing_ratio, dtype=float)
    temperature = np.asarray(temperature_k, dtype=float)
    pressure = np.asarray(pressure_pa, dtype=float)
    qv_sat = rv_saturation(temperature, pressure)
    dp = -np.gradient(pressure, axis=0)
    within_column = (pressure < 1200.0e2) & (pressure >= 0.0)
    valid = within_column & np.isfinite(qv) & np.isfinite(qv_sat) & np.isfinite(dp)
    pw = np.sum(np.where(valid, qv * dp / G, 0.0), axis=0)
    pw_sat = np.sum(np.where(valid, qv_sat * dp / G, 0.0), axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        crh = pw / pw_sat
    return np.where((pw_sat > 0) & np.isfinite(crh), crh, np.nan)


def partition_sam_condensate(
    temperature_k: np.ndarray, qn: np.ndarray, qp: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Legacy SAM phase partition used by ``precip_eff_rce.ipynb``.

    The snow/graupel expressions are preserved verbatim for pclass compatibility,
    even though their sum with rain double-counts QP.
    """

    temperature = np.asarray(temperature_k, dtype=float)
    qn_values = np.asarray(qn, dtype=float)
    qp_values = np.asarray(qp, dtype=float)
    cloud_fraction = np.clip((temperature - 253.16) / 20.0, 0.0, 1.0)
    rain_fraction = np.clip((temperature - 268.16) / 15.0, 0.0, 1.0)
    graupel_fraction = np.clip((temperature - 223.16) / 60.0, 0.0, 1.0)
    qcloud = cloud_fraction * qn_values
    qice = (1.0 - cloud_fraction) * qn_values
    qrain = rain_fraction * qp_values
    qsnow = (1.0 - rain_fraction * (1.0 - graupel_fraction)) * qp_values
    qgraupel = (1.0 - rain_fraction * graupel_fraction) * qp_values
    return qcloud, qice, qrain, qsnow, qgraupel


def classify_integrated_hydrometeors(q_int: np.ndarray) -> np.ndarray:
    """Classify integrated hydrometeors into primitive SAM class codes 0--5."""

    integrated = np.asarray(q_int, dtype=float)
    if integrated.shape[0] != 5:
        raise ValueError("q_int must contain cloud, rain, ice, snow, and graupel")
    lwp = integrated[0] + integrated[1]
    iwp = integrated[2] + integrated[3] + integrated[4]
    twp = lwp + iwp
    with np.errstate(divide="ignore", invalid="ignore"):
        cloud_ratio = iwp / lwp

    result = np.zeros(lwp.shape, dtype=np.uint8)
    cloudy = (lwp != 0) & (twp > 1.0e-1)
    convective = cloudy & (cloud_ratio <= 2.0)
    stratiform_family = cloudy & (cloud_ratio > 2.0)
    result[
        convective & (integrated[1] >= 1.0e-1) & (integrated[4] >= 1.0e-4)
    ] = 1
    result[
        convective & (integrated[1] >= 1.0e-1) & (integrated[4] < 1.0e-4)
    ] = 2
    result[convective & (integrated[1] < 1.0e-1)] = 3
    result[stratiform_family & (integrated[1] >= 1.0e-2)] = 4
    result[stratiform_family & (integrated[1] < 1.0e-2)] = 5

    invalid = ~np.all(np.isfinite(integrated), axis=0)
    result[invalid] = PCLASS_FILL_VALUE
    return result


def generate_pclass(
    temperature_k: np.ndarray,
    qn: np.ndarray,
    qp: np.ndarray,
    pressure_pa: np.ndarray,
) -> np.ndarray:
    """Generate pclass from raw SAM condensate using the legacy algorithm."""

    hydrometeors = partition_sam_condensate(temperature_k, qn, qp)
    dp = -np.gradient(np.asarray(pressure_pa, dtype=float), axis=0)
    q_integrated = np.array(
        [np.sum(field * dp / G, axis=0) for field in hydrometeors], dtype=float
    )
    return classify_integrated_hydrometeors(q_integrated)


def validate_pclass(values: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Validate and normalize cached pclass labels to uint8."""

    array = np.asarray(values)
    if array.shape != shape:
        raise CacheValidationError(
            f"pclass shape {array.shape} does not match expected {shape}"
        )
    finite = np.isfinite(array)
    if np.any(finite & ((array < 0) | (array > 5) | (array != np.floor(array)))):
        raise CacheValidationError("pclass contains values outside primitive codes 0--5")
    output = np.full(shape, PCLASS_FILL_VALUE, dtype=np.uint8)
    output[finite] = array[finite].astype(np.uint8)
    return output


def lw_acre_from_heating(
    delta_heating_k_day: np.ndarray,
    pressure_hpa: np.ndarray = PRESSURE_HPA,
    require_complete_column: bool = True,
) -> np.ndarray:
    """Integrate a pressure-resolved LW heating field to W m-2."""

    heating = np.asarray(delta_heating_k_day, dtype=float)
    pressure = np.asarray(pressure_hpa, dtype=float)
    if heating.shape[0] != pressure.size:
        raise ValueError("heating's first dimension must match pressure_hpa")
    order = np.argsort(pressure)
    pressure_pa = pressure[order] * 100.0
    sorted_heating = heating[order] / SECONDS_PER_DAY
    flattened = sorted_heating.reshape(sorted_heating.shape[0], -1)
    result = np.full(flattened.shape[1], np.nan, dtype=float)
    for icol in range(flattened.shape[1]):
        column = flattened[:, icol]
        finite = np.isfinite(column) & np.isfinite(pressure_pa)
        if require_complete_column and not np.all(finite):
            continue
        if np.count_nonzero(finite) < 2:
            continue
        result[icol] = (CP / G) * np.trapz(column[finite], pressure_pa[finite])
    return result.reshape(heating.shape[1:])


def global_class_area_percent(class_counts: np.ndarray) -> np.ndarray:
    """WRF-style percent of all valid samples belonging to each class/bin."""

    counts = np.asarray(class_counts, dtype=float)
    total = np.nansum(counts)
    if total <= 0:
        return np.full(counts.shape, np.nan)
    return 100.0 * counts / total


def _resolve_3d_directory(root: Path) -> Path:
    direct = root / "NC_files" / "OUT_3D"
    if direct.is_dir():
        return direct
    if root.is_dir() and root.name == "OUT_3D":
        return root
    raise FileNotFoundError(f"Could not find NC_files/OUT_3D below {root}")


def _resolve_2d_directory(root: Path) -> Path:
    direct = root / "NC_files" / "OUT_2D"
    if direct.is_dir():
        return direct
    if root.is_dir() and root.name == "OUT_2D":
        return root
    raise FileNotFoundError(f"Could not find NC_files/OUT_2D below {root}")


def _filename_time_hint(path: Path) -> float | None:
    match = re.search(r"(\d{10})(?=\.nc$)", path.name)
    if match is None:
        return None
    return int(match.group(1)) / (300.0 * 24.0)


def _read_scalar_time(path: Path) -> float:
    try:
        with xr.open_dataset(path, decode_times=False) as dataset:
            value = np.asarray(dataset["time"].values).squeeze()
    except Exception as exc:  # pragma: no cover - exercised with real files
        raise RCEProcessingError(f"Could not read scalar time from {path}: {exc}") from exc
    if np.ndim(value) != 0 or not np.isfinite(value):
        raise RCEProcessingError(f"Expected one finite scalar time in {path}")
    return float(value)


def discover_3d_records(
    root: Path | str,
    start_day: float,
    end_day: float,
    target_times: Sequence[float] | None = None,
    tolerance: float = 1.0e-6,
) -> list[FileRecord]:
    """Discover and validate single-time 3-D files in a half-open window."""

    directory = _resolve_3d_directory(Path(root))
    paths = sorted(directory.glob("*.nc"))
    if not paths:
        raise FileNotFoundError(f"No NetCDF files found in {directory}")

    target = None if target_times is None else np.asarray(target_times, dtype=float)
    records: list[FileRecord] = []
    for path in paths:
        hint = _filename_time_hint(path)
        if hint is not None and not (start_day - tolerance <= hint < end_day + tolerance):
            continue
        if target is not None and hint is not None:
            if not np.any(np.isclose(hint, target, atol=tolerance, rtol=0.0)):
                continue
        actual_time = _read_scalar_time(path)
        if not (start_day <= actual_time < end_day):
            continue
        if target is not None and not np.any(
            np.isclose(actual_time, target, atol=tolerance, rtol=0.0)
        ):
            continue
        records.append(FileRecord(actual_time, path))

    records.sort(key=lambda record: record.time)
    keys: set[int] = set()
    for record in records:
        key = _time_key(record.time, tolerance)
        if key in keys:
            raise RCEProcessingError(f"Duplicate 3-D time {record.time} below {root}")
        keys.add(key)
    return records


def validate_sampling_cadence(
    records: Sequence[FileRecord],
    start_day: int,
    end_day: int,
    samples_per_day: int = 4,
) -> None:
    """Require the expected number of CTL samples in every requested day."""

    errors: list[str] = []
    for day in range(start_day, end_day):
        day_records = [record for record in records if day <= record.time < day + 1]
        if len(day_records) != samples_per_day:
            errors.append(
                f"day {day}: expected {samples_per_day}, found {len(day_records)} "
                f"({[record.time for record in day_records]})"
            )
    if errors:
        raise RCEProcessingError("Incomplete six-hour sampling:\n" + "\n".join(errors))


def select_regular_3d_records(
    records: Sequence[FileRecord],
    start_day: int,
    end_day: int,
    samples_per_day: int = 4,
    tolerance: float = 1.0e-6,
) -> tuple[list[FileRecord], list[FileRecord]]:
    """Select records on the regular within-day 3-D sampling phases.

    The standard RCE 3-D cadence is every six hours.  Some output windows also
    contain intervening hourly 3-D files; those records are excluded so each
    day receives equal weight in the daily statistics.
    """

    phases = np.arange(samples_per_day, dtype=float) / samples_per_day
    selected: list[FileRecord] = []
    excluded: list[FileRecord] = []
    for record in records:
        day = math.floor(record.time)
        if not start_day <= day < end_day:
            continue
        expected = day + phases
        if np.any(np.isclose(record.time, expected, atol=tolerance, rtol=0.0)):
            selected.append(record)
        else:
            excluded.append(record)
    return selected, excluded


def group_records_by_day(
    records: Sequence[FileRecord], start_day: int, end_day: int
) -> dict[int, list[FileRecord]]:
    return {
        day: [record for record in records if day <= record.time < day + 1]
        for day in range(start_day, end_day)
    }


def _squeezed_array(dataset: xr.Dataset, name: str, ndim: int) -> np.ndarray:
    if name not in dataset:
        raise RCEProcessingError(f"Required variable {name!r} is absent")
    variable = dataset[name]
    # Drop singleton record dimensions (normally ``time``) but preserve singleton
    # physical dimensions so small synthetic/test domains remain valid.
    for dimension in list(variable.dims):
        if variable.ndim <= ndim:
            break
        if variable.sizes[dimension] == 1:
            variable = variable.isel({dimension: 0}, drop=True)
    values = np.asarray(variable.values)
    if values.ndim != ndim:
        raise RCEProcessingError(
            f"{name} has {values.ndim} dimensions after squeeze; expected {ndim}"
        )
    if not np.issubdtype(values.dtype, np.number):
        raise RCEProcessingError(f"{name} must be numeric")
    return values.astype(np.float64, copy=False)


def _normalize_unit_text(units: str) -> str:
    return (
        units.lower()
        .replace(" ", "")
        .replace("**", "^")
        .replace("−", "-")
        .replace("⁻", "-")
    )


def heating_to_k_day(
    values: np.ndarray, units: str | None, override: str | None = None
) -> tuple[np.ndarray, str]:
    """Normalize a temperature tendency to K/day using a strict unit whitelist."""

    original = override or units
    if not original:
        raise RCEProcessingError(
            "LW-heating units are missing; supply --lw-units explicitly"
        )
    normalized = _normalize_unit_text(original)
    per_day = {
        "k/day",
        "k/d",
        "kday-1",
        "kd-1",
        "kday^-1",
        "kd^-1",
    }
    per_second = {"k/s", "ks-1", "ks^-1", "k/sec", "ksecond-1"}
    per_hour = {"k/hour", "k/hr", "kh-1", "kh^-1", "khour-1"}
    if normalized in per_day:
        factor = 1.0
    elif normalized in per_second:
        factor = SECONDS_PER_DAY
    elif normalized in per_hour:
        factor = 24.0
    else:
        raise RCEProcessingError(
            f"Unsupported LW-heating units {original!r}; expected K/day or K/s"
        )
    return np.asarray(values, dtype=float) * factor, str(original)


def rain_to_mm_day(
    values: np.ndarray, units: str | None, override: str | None = None
) -> tuple[np.ndarray, str]:
    """Normalize precipitation rate to mm/day."""

    original = override or units
    if not original:
        raise RCEProcessingError("rain units are missing; supply --rain-units")
    normalized = _normalize_unit_text(original)
    mm_day = {"mm/day", "mm/d", "mmday-1", "mmd-1", "mmday^-1", "mmd^-1"}
    mm_hour = {"mm/hour", "mm/hr", "mmh-1", "mmh^-1"}
    mass_flux = {
        "kgm-2s-1",
        "kg/m^2/s",
        "kgm^-2s^-1",
        "kgm-2sec-1",
    }
    if normalized in mm_day:
        factor = 1.0
    elif normalized in mm_hour:
        factor = 24.0
    elif normalized in mass_flux:
        factor = SECONDS_PER_DAY
    else:
        raise RCEProcessingError(
            f"Unsupported precipitation units {original!r}; supply --rain-units"
        )
    return np.asarray(values, dtype=float) * factor, str(original)


def flux_to_w_m2(
    values: np.ndarray, units: str | None, override: str | None = None
) -> tuple[np.ndarray, str]:
    """Validate a radiative flux expressed in W m-2."""

    original = override or units
    if not original:
        raise RCEProcessingError("LW-flux units are missing")
    normalized = _normalize_unit_text(original)
    accepted = {"w/m2", "w/m^2", "wm-2", "wm^-2", "watt/m2", "watts/m2"}
    if normalized not in accepted:
        raise RCEProcessingError(
            f"Unsupported LW-flux units {original!r}; expected W m-2"
        )
    return np.asarray(values, dtype=float), str(original)


def lw_acre_from_fluxes(
    lwns: np.ndarray,
    lwnt: np.ndarray,
    lwnsc: np.ndarray,
    lwntc: np.ndarray,
) -> np.ndarray:
    """Return local LW-ACRE from net all-sky and clear-sky boundary fluxes.

    The SAM net-flux convention used here is upward-positive.  Positive output
    therefore means clouds increase atmospheric longwave flux convergence.
    """

    fields = [np.asarray(value, dtype=float) for value in (lwns, lwnt, lwnsc, lwntc)]
    shapes = {field.shape for field in fields}
    if len(shapes) != 1:
        raise ValueError("LW-ACRE flux fields must have identical shapes")
    return (fields[0] - fields[1]) - (fields[2] - fields[3])


def numeric_3d_candidates(dataset: xr.Dataset) -> list[dict[str, str]]:
    """Describe numeric three-dimensional variables for preflight errors."""

    candidates: list[dict[str, str]] = []
    for name, variable in dataset.data_vars.items():
        squeezed = variable.squeeze(drop=True)
        if squeezed.ndim == 3 and np.issubdtype(variable.dtype, np.number):
            candidates.append(
                {
                    "name": name,
                    "units": str(variable.attrs.get("units", "")),
                    "long_name": str(variable.attrs.get("long_name", "")),
                    "dims": str(tuple(squeezed.dims)),
                }
            )
    return candidates


def detect_lw_heating_variable(
    dataset: xr.Dataset, requested: str | None = None
) -> str:
    """Resolve a genuine LW-only 3-D tendency by explicit name or allowlist."""

    name_lookup = {name.upper(): name for name in dataset.data_vars}
    if requested is not None:
        if requested.upper() == "QRAD":
            raise RCEProcessingError(
                "QRAD is total radiative heating and cannot be used as LW heating"
            )
        if requested not in dataset and requested.upper() not in name_lookup:
            raise RCEProcessingError(
                f"Requested LW variable {requested!r} is absent. "
                f"3-D candidates: {numeric_3d_candidates(dataset)}"
            )
        resolved = dataset.data_vars.get(requested)
        name = requested if resolved is not None else name_lookup[requested.upper()]
        _squeezed_array(dataset, name, 3)
        return name

    matches = [name_lookup[name] for name in LW_HEATING_ALLOWLIST if name in name_lookup]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise RCEProcessingError(
            "No allowlisted LW-only 3-D heating variable was found. Supply --lw-var. "
            f"Candidates: {numeric_3d_candidates(dataset)}"
        )
    raise RCEProcessingError(
        f"Multiple LW-heating candidates {matches}; choose one with --lw-var"
    )


def _pressure_units_factor(
    variable: xr.DataArray, values: np.ndarray, default_factor: float | None = None
) -> float:
    units = _normalize_unit_text(str(variable.attrs.get("units", "")))
    if units in {"pa", "pascal", "pascals"}:
        return 1.0
    if units in {"hpa", "mb", "mbar", "millibar"}:
        return 100.0
    if default_factor is not None:
        return default_factor
    finite = np.asarray(values)[np.isfinite(values)]
    if finite.size == 0:
        raise RCEProcessingError(f"Pressure variable {variable.name} has no finite data")
    return 100.0 if np.nanmedian(np.abs(finite)) < 2000.0 else 1.0


def pressure_from_dataset(dataset: xr.Dataset) -> np.ndarray:
    """Return full SAM pressure ``p + PP`` in Pa."""

    base = _squeezed_array(dataset, "p", 1)
    perturbation = _squeezed_array(dataset, "PP", 3)
    # SAM convention (also used in precip_eff_rce.ipynb): p is hPa and PP is Pa
    # when unit attributes are absent.
    base_pa = base * _pressure_units_factor(dataset["p"], base, default_factor=100.0)
    pp_pa = perturbation * _pressure_units_factor(
        dataset["PP"], perturbation, default_factor=1.0
    )
    if pp_pa.shape[0] != base_pa.size:
        raise RCEProcessingError("p and PP vertical dimensions do not match")
    return base_pa[:, None, None] + pp_pa


def mixing_ratio_from_dataset(dataset: xr.Dataset, name: str = "QV") -> np.ndarray:
    values = _squeezed_array(dataset, name, 3)
    units = _normalize_unit_text(str(dataset[name].attrs.get("units", "")))
    if "g/kg" in units or "gkg-1" in units or "gkg^-1" in units:
        return values * 1.0e-3
    finite = np.abs(values[np.isfinite(values)])
    if finite.size and np.nanmedian(finite) > 0.2:
        return values * 1.0e-3
    return values


def xy_coordinates(dataset: xr.Dataset) -> tuple[np.ndarray, np.ndarray]:
    if "x" not in dataset.coords or "y" not in dataset.coords:
        raise RCEProcessingError("3-D datasets must contain one-dimensional x/y coordinates")
    x = np.asarray(dataset.coords["x"].values).squeeze()
    y = np.asarray(dataset.coords["y"].values).squeeze()
    if x.ndim != 1 or y.ndim != 1:
        raise RCEProcessingError("x and y coordinates must be one-dimensional")
    return x.astype(float), y.astype(float)


class LegacyDiagnosticsCache:
    """Safely resolve CRH/pclass from legacy notebook pickles when possible."""

    def __init__(self, pickle_dir: Path | str, tolerance: float = 1.0e-6):
        self.pickle_dir = Path(pickle_dir)
        self.tolerance = tolerance
        self._payloads: dict[Path, dict] = {}
        self._coords: dict | None | bool = False
        self._candidates = sorted(
            path
            for path in self.pickle_dir.glob("RCE_days_*.pickle")
            if not path.name.endswith("_2d.pickle")
        )

    def _load_coords(self) -> dict | None:
        if self._coords is not False:
            return self._coords if isinstance(self._coords, dict) else None
        path = self.pickle_dir / "exper_coords_RCE.pickle"
        if not path.exists():
            self._coords = None
            return None
        with path.open("rb") as handle:
            payload = pickle.load(handle)
        self._coords = payload
        return payload

    def _grid_matches(self, x: np.ndarray, y: np.ndarray) -> bool:
        coords = self._load_coords()
        if coords is None or "x" not in coords or "y" not in coords:
            return False
        stored_x = np.asarray(coords["x"], dtype=float)
        stored_y = np.asarray(coords["y"], dtype=float)
        raw_x = np.asarray(x, dtype=float)
        raw_y = np.asarray(y, dtype=float)
        if raw_x.shape != stored_x.shape or raw_y.shape != stored_y.shape:
            return False
        # The notebook stores x/y in km while raw SAM coordinates are metres.
        if np.nanmax(np.abs(raw_x)) > 10.0 * max(np.nanmax(np.abs(stored_x)), 1.0):
            raw_x = raw_x / 1000.0
            raw_y = raw_y / 1000.0
        return bool(
            np.allclose(raw_x, stored_x, atol=1.0e-6, rtol=1.0e-8)
            and np.allclose(raw_y, stored_y, atol=1.0e-6, rtol=1.0e-8)
        )

    def _load(self, path: Path) -> dict:
        if path not in self._payloads:
            with path.open("rb") as handle:
                self._payloads[path] = pickle.load(handle)
        return self._payloads[path]

    def get(
        self,
        experiment: str,
        variable: str,
        time: float,
        shape: tuple[int, int],
        x: np.ndarray,
        y: np.ndarray,
        source_path: Path | str,
    ) -> np.ndarray | None:
        if not self._grid_matches(x, y):
            return None
        suffix = "_rh" if experiment.lower() == "rh" else ""
        times_key = f"times3d{suffix}"
        files_key = f"files3d{suffix}"
        vars_key = f"vars3d{suffix}"
        source_name = Path(source_path).name
        for path in self._candidates:
            payload = self._load(path)
            if times_key not in payload or vars_key not in payload:
                continue
            variables = payload[vars_key]
            if variable not in variables:
                continue
            times = np.asarray(payload[times_key], dtype=float)
            matches = np.flatnonzero(
                np.isclose(times, time, atol=self.tolerance, rtol=0.0)
            )
            if matches.size != 1:
                continue
            index = int(matches[0])
            cached_files = payload.get(files_key)
            if cached_files is None or index >= len(cached_files):
                continue
            if Path(cached_files[index]).name != source_name:
                # Reject the known legacy subset bug instead of trusting array order.
                continue
            values = np.asarray(variables[variable][index])
            if values.shape != shape:
                continue
            return values.copy()
        return None


def load_pclass_sidecar(
    path: Path | str,
    requested_times: Sequence[float],
    x: np.ndarray,
    y: np.ndarray,
    source_files: Mapping[int, Path] | None = None,
    tolerance: float = 1.0e-6,
) -> dict[int, np.ndarray]:
    """Load safely aligned pclass labels from a daily sidecar."""

    cache_path = Path(path)
    if not cache_path.exists():
        return {}
    with xr.open_dataset(cache_path, mask_and_scale=False) as dataset:
        if dataset.attrs.get("classifier_version") != CLASSIFIER_VERSION:
            raise CacheValidationError(f"Unsupported classifier in {cache_path}")
        if not np.allclose(dataset["x"].values, x) or not np.allclose(
            dataset["y"].values, y
        ):
            raise CacheValidationError(f"pclass grid mismatch in {cache_path}")
        times = np.asarray(dataset["time"].values, dtype=float)
        files = (
            [str(value) for value in dataset["source_file"].values]
            if "source_file" in dataset
            else None
        )
        result: dict[int, np.ndarray] = {}
        for requested in requested_times:
            matches = np.flatnonzero(
                np.isclose(times, requested, atol=tolerance, rtol=0.0)
            )
            if matches.size == 0:
                continue
            if matches.size != 1:
                raise CacheValidationError(f"Duplicate time {requested} in {cache_path}")
            index = int(matches[0])
            key = _time_key(requested, tolerance)
            if source_files is not None and files is not None:
                expected = source_files.get(key)
                if expected is not None and Path(files[index]).name != expected.name:
                    raise CacheValidationError(
                        f"pclass source mismatch at time {requested} in {cache_path}"
                    )
            result[key] = validate_pclass(
                dataset["pclass"].isel(time=index).values, (len(y), len(x))
            )
        return result


def write_pclass_sidecar(
    path: Path | str,
    times: Sequence[float],
    values: Sequence[np.ndarray],
    x: np.ndarray,
    y: np.ndarray,
    source_files: Sequence[Path | str],
) -> Path:
    """Atomically write compressed uint8 pclass labels for one day."""

    pclass = np.stack([validate_pclass(value, (len(y), len(x))) for value in values])
    dataset = xr.Dataset(
        data_vars={
            "pclass": (("time", "y", "x"), pclass),
            "source_file": (("time",), np.asarray([str(path) for path in source_files])),
        },
        coords={"time": np.asarray(times, dtype=float), "x": x, "y": y},
        attrs={
            "schema_version": SCHEMA_VERSION,
            "classifier_version": CLASSIFIER_VERSION,
            "description": "Primitive SAM precipitation class labels (0-5)",
            "compatibility_note": "Legacy precip_eff_rce condensate partition preserved",
        },
    )
    dataset["pclass"].attrs.update(
        long_name="precipitation/cloud classification",
        valid_min=np.uint8(0),
        valid_max=np.uint8(5),
    )
    return atomic_to_netcdf(
        dataset,
        Path(path),
        encoding={"pclass": {"zlib": True, "complevel": 2, "_FillValue": 255}},
    )


class TimeIndexed2DFields:
    """Index SAM 2-D output once and read one or more fields by model time."""

    def __init__(
        self,
        root: Path | str,
        start_day: float,
        end_day: float,
        tolerance: float = 1.0e-6,
    ):
        self.tolerance = tolerance
        directory = _resolve_2d_directory(Path(root))
        paths = sorted(directory.glob("*.nc"))
        if not paths:
            raise FileNotFoundError(f"No 2-D NetCDF files found in {directory}")
        self._index: dict[int, tuple[Path, int, float]] = {}
        for path in paths:
            with xr.open_dataset(path, decode_times=False) as dataset:
                if "time" not in dataset:
                    continue
                times = np.asarray(dataset["time"].values, dtype=float).ravel()
            for index, time in enumerate(times):
                if not (start_day <= time < end_day):
                    continue
                key = _time_key(time, tolerance)
                if key in self._index:
                    raise RCEProcessingError(f"Duplicate 2-D time {time}")
                self._index[key] = (path, index, float(time))

    def read(
        self, time: float, variables: Sequence[str]
    ) -> tuple[dict[str, np.ndarray], dict[str, str | None], np.ndarray, np.ndarray]:
        """Read named two-dimensional fields from one exact decoded time."""

        key = _time_key(time, self.tolerance)
        if key not in self._index:
            raise RCEProcessingError(f"No 2-D record at time {time}")
        path, index, stored_time = self._index[key]
        if not np.isclose(stored_time, time, atol=self.tolerance, rtol=0.0):
            raise RCEProcessingError(f"2-D time mismatch for requested time {time}")
        with xr.open_dataset(path, decode_times=False) as dataset:
            values_by_name: dict[str, np.ndarray] = {}
            units_by_name: dict[str, str | None] = {}
            for name in variables:
                if name not in dataset:
                    raise RCEProcessingError(f"{name} is absent from {path}")
                variable = dataset[name]
                if "time" not in variable.dims:
                    raise RCEProcessingError(f"{name} in {path} has no time dimension")
                time_dim = "time"
                values = np.asarray(variable.isel({time_dim: index}).squeeze().values)
                if values.ndim != 2:
                    raise RCEProcessingError(f"{name} at {time} is not two-dimensional")
                if not np.any(np.isfinite(values)):
                    raise RCEProcessingError(f"{name} at {time} contains no finite values")
                values_by_name[name] = values.astype(float, copy=False)
                units_by_name[name] = variable.attrs.get("units")
            x, y = xy_coordinates(dataset)
        return values_by_name, units_by_name, x, y

    def read_rain(
        self, time: float, variable: str, units_override: str | None = None
    ) -> tuple[np.ndarray, str, np.ndarray, np.ndarray]:
        values, units, x, y = self.read(time, (variable,))
        normalized, original_units = rain_to_mm_day(
            values[variable], units[variable], units_override
        )
        return normalized, original_units, x, y

    def read_lw_acre(
        self, time: float
    ) -> tuple[np.ndarray, dict[str, str], np.ndarray, np.ndarray]:
        """Read and validate the four SAM fluxes needed for local LW-ACRE."""

        values, units, x, y = self.read(time, LW_ACRE_FLUX_VARIABLES)
        normalized: dict[str, np.ndarray] = {}
        original_units: dict[str, str] = {}
        for name in LW_ACRE_FLUX_VARIABLES:
            normalized[name], original_units[name] = flux_to_w_m2(
                values[name], units[name]
            )
        acre = lw_acre_from_fluxes(
            normalized["LWNS"],
            normalized["LWNT"],
            normalized["LWNSC"],
            normalized["LWNTC"],
        )
        return acre, original_units, x, y


class TimeIndexed2DField:
    """Compatibility wrapper for a single precipitation field."""

    def __init__(
        self,
        root: Path | str,
        variable: str,
        start_day: float,
        end_day: float,
        tolerance: float = 1.0e-6,
    ):
        self.variable = variable
        self._reader = TimeIndexed2DFields(root, start_day, end_day, tolerance)

    def read(
        self, time: float, units_override: str | None = None
    ) -> tuple[np.ndarray, str, np.ndarray, np.ndarray]:
        return self._reader.read_rain(time, self.variable, units_override)


def validate_lw_acre_inputs(
    config: PreprocessConfig, records: Sequence[FileRecord]
) -> dict[str, str]:
    """Validate direct CTL LW-ACRE inputs against every matched 3-D record."""

    reader = TimeIndexed2DFields(
        config.ctl_root, config.start_day, config.end_day, config.time_tolerance
    )
    units_seen: dict[str, str] = {}
    for record in records:
        _, units, flux_x, flux_y = reader.read_lw_acre(record.time)
        with xr.open_dataset(record.path, decode_times=False) as dataset:
            ctl_x, ctl_y = xy_coordinates(dataset)
        if not np.allclose(ctl_x, flux_x) or not np.allclose(ctl_y, flux_y):
            raise RCEProcessingError("3-D and LW-ACRE flux grids differ")
        for name, current in units.items():
            previous = units_seen.get(name)
            if previous is not None and previous != current:
                raise RCEProcessingError(
                    f"{name} units change across requested records: "
                    f"{previous!r} versus {current!r}"
                )
            units_seen[name] = current
    return units_seen


def _daily_accumulator() -> dict[str, np.ndarray]:
    npressure = PRESSURE_HPA.size
    nbins = CRH_EDGES.size - 1
    return {
        "lw_heating_sum": np.zeros((npressure, nbins), dtype=np.float64),
        "lw_heating_count": np.zeros((npressure, nbins), dtype=np.int64),
        "crh_sample_count": np.zeros(nbins, dtype=np.int64),
        "mass_flux_sum": np.zeros((npressure, nbins), dtype=np.float64),
        "mass_flux_count": np.zeros((npressure, nbins), dtype=np.int64),
        "rain_sum": np.zeros(nbins, dtype=np.float64),
        "rain_count": np.zeros(nbins, dtype=np.int64),
        "lw_acre_sum": np.zeros(nbins, dtype=np.float64),
        "lw_acre_count": np.zeros(nbins, dtype=np.int64),
        "pclass_count": np.zeros((6, nbins), dtype=np.int64),
    }


def _daily_dataset(
    accumulator: Mapping[str, np.ndarray],
    day: int,
    records: Sequence[FileRecord],
    lw_var: str,
    lw_original_units: str,
    rain_original_units: str,
    pclass_sources: Mapping[str, int],
    lw_acre_flux_units: Mapping[str, str],
    rain_var: str = "Prec",
) -> xr.Dataset:
    edges = CRH_EDGES
    dataset = xr.Dataset(
        data_vars={
            "lw_heating_sum": (
                ("pressure_hpa", "crh_bin"),
                accumulator["lw_heating_sum"],
            ),
            "lw_heating_count": (
                ("pressure_hpa", "crh_bin"),
                accumulator["lw_heating_count"],
            ),
            "crh_sample_count": (
                ("crh_bin",),
                accumulator["crh_sample_count"],
            ),
            "mass_flux_sum": (
                ("pressure_hpa", "crh_bin"),
                accumulator["mass_flux_sum"],
            ),
            "mass_flux_count": (
                ("pressure_hpa", "crh_bin"),
                accumulator["mass_flux_count"],
            ),
            "rain_sum": (("crh_bin",), accumulator["rain_sum"]),
            "rain_count": (("crh_bin",), accumulator["rain_count"]),
            "ctl_lw_acre_sum": (("crh_bin",), accumulator["lw_acre_sum"]),
            "ctl_lw_acre_count": (("crh_bin",), accumulator["lw_acre_count"]),
            "pclass_count": (
                ("class_code", "crh_bin"),
                accumulator["pclass_count"],
            ),
            "crh_bin_center": (
                ("crh_bin",), 0.5 * (edges[:-1] + edges[1:])
            ),
        },
        coords={
            "pressure_hpa": PRESSURE_HPA,
            "crh_bin": edges[:-1],
            "crh_edge": edges,
            "class_code": CLASS_CODES,
            "class_name": (("class_code",), CLASS_NAMES),
        },
        attrs={
            "schema_version": SCHEMA_VERSION,
            "day": int(day),
            "time_bounds": f"[{day}, {day + 1})",
            "sample_times": json.dumps([record.time for record in records]),
            "ctl_source_files": json.dumps([str(record.path) for record in records]),
            "lw_variable": lw_var,
            "lw_ctl_original_units": lw_original_units,
            "rain_original_units": rain_original_units,
            "rain_variable": rain_var,
            "lw_acre_flux_variables": json.dumps(LW_ACRE_FLUX_VARIABLES),
            "lw_acre_flux_units": json.dumps(dict(lw_acre_flux_units), sort_keys=True),
            "lw_acre_definition": "(LWNS-LWNT)-(LWNSC-LWNTC)",
            "pclass_sources": json.dumps(dict(pclass_sources), sort_keys=True),
            "classifier_version": CLASSIFIER_VERSION,
            "bin_convention": "lower-inclusive, upper-exclusive",
            "sampling_selection": "regular six-hourly CTL 3-D records only",
            "lw_anomaly_definition": "LW_CTL(z,y,x,t) - mean_domain(LW_CTL(z,:, :,t))",
            "heating_diagnostic_label": "CTL local longwave heating anomaly",
        },
    )
    dataset["lw_heating_sum"].attrs["units"] = "K day-1 summed over samples"
    dataset["mass_flux_sum"].attrs["units"] = "kg m-2 s-1 summed over samples"
    dataset["rain_sum"].attrs["units"] = "mm day-1 summed over samples"
    dataset["ctl_lw_acre_sum"].attrs["units"] = "W m-2 summed over samples"
    dataset["pressure_hpa"].attrs["units"] = "hPa"
    dataset["crh_bin"].attrs.update(
        long_name="left edge of column relative humidity bin", units="1"
    )
    return dataset


def _numeric_encoding(dataset: xr.Dataset) -> dict[str, dict]:
    encoding: dict[str, dict] = {}
    for name, variable in dataset.data_vars.items():
        if np.issubdtype(variable.dtype, np.number):
            encoding[name] = {"zlib": True, "complevel": 2}
    return encoding


def atomic_to_netcdf(
    dataset: xr.Dataset,
    path: Path | str,
    encoding: Mapping[str, Mapping] | None = None,
) -> Path:
    """Write a NetCDF file atomically in its destination directory."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        prefix=destination.name + ".", suffix=".tmp", dir=destination.parent, delete=False
    )
    temporary = Path(handle.name)
    handle.close()
    engine = "netcdf4" if importlib.util.find_spec("netCDF4") else None
    try:
        kwargs = {"encoding": dict(encoding or {})}
        if engine is not None:
            kwargs["engine"] = engine
        dataset.to_netcdf(temporary, **kwargs)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination


def _validate_existing_daily(
    path: Path,
    day: int,
    lw_var: str,
    config: PreprocessConfig,
) -> None:
    with xr.open_dataset(path) as dataset:
        if dataset.attrs.get("schema_version") != SCHEMA_VERSION:
            raise CacheValidationError(f"Unsupported schema in {path}")
        if int(dataset.attrs.get("day", -1)) != day:
            raise CacheValidationError(f"Day mismatch in {path}")
        if dataset.attrs.get("lw_variable") != lw_var:
            raise CacheValidationError(
                f"{path} uses {dataset.attrs.get('lw_variable')}, "
                f"not {lw_var}"
            )
        if dataset.attrs.get("rain_variable", "Prec") != config.rain_var:
            raise CacheValidationError(
                f"{path} uses rain variable {dataset.attrs.get('rain_variable')!r}, "
                f"not {config.rain_var!r}"
            )
        if config.lw_units is not None:
            cached_units = dataset.attrs.get("lw_ctl_original_units")
            if cached_units != config.lw_units:
                raise CacheValidationError(
                    f"{path} used LW units {cached_units!r}, not override "
                    f"{config.lw_units!r}"
                )
        if config.rain_units is not None and dataset.attrs.get(
            "rain_original_units"
        ) != config.rain_units:
            raise CacheValidationError(
                f"{path} used rain units {dataset.attrs.get('rain_original_units')!r}, "
                f"not override {config.rain_units!r}"
            )
        if not np.array_equal(dataset["pressure_hpa"].values, PRESSURE_HPA) or not np.array_equal(
            dataset["crh_edge"].values, CRH_EDGES
        ):
            raise CacheValidationError(f"Pressure or CRH coordinates differ in {path}")


def _read_lw_heating(
    dataset: xr.Dataset, variable: str, units_override: str | None
) -> tuple[np.ndarray, str]:
    values = _squeezed_array(dataset, variable, 3)
    converted, original_units = heating_to_k_day(
        values, dataset[variable].attrs.get("units"), units_override
    )
    if not np.any(np.isfinite(converted)):
        raise RCEProcessingError(f"{variable} contains no finite values")
    return converted, original_units


def lw_heating_anomaly(values: np.ndarray) -> np.ndarray:
    """Subtract the finite horizontal mean from each LW level and output time."""

    field = np.asarray(values, dtype=float)
    if field.ndim != 3:
        raise ValueError("LW heating anomaly requires a (z, y, x) field")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        domain_mean = np.nanmean(field, axis=(1, 2), keepdims=True)
    return field - domain_mean


def _preflight(
    config: PreprocessConfig,
    require_lw_heating: bool = True,
) -> tuple[list[FileRecord], str, dict[str, object]]:
    ctl_records = discover_3d_records(
        config.ctl_root,
        config.start_day,
        config.end_day,
        tolerance=config.time_tolerance,
    )
    ctl_records, excluded_ctl_records = select_regular_3d_records(
        ctl_records,
        config.start_day,
        config.end_day,
        config.samples_per_day,
        config.time_tolerance,
    )
    validate_sampling_cadence(
        ctl_records, config.start_day, config.end_day, config.samples_per_day
    )
    with xr.open_dataset(ctl_records[0].path, decode_times=False) as ctl_dataset:
        base_inventory = {
            "window": [config.start_day, config.end_day],
            "ctl_times": [record.time for record in ctl_records],
            "excluded_ctl_3d_times": [record.time for record in excluded_ctl_records],
            "ctl_3d_candidates": numeric_3d_candidates(ctl_dataset),
        }
        if not require_lw_heating:
            return ctl_records, "", base_inventory
        try:
            lw_var = detect_lw_heating_variable(ctl_dataset, config.lw_var)
            ctl_values, ctl_units = _read_lw_heating(
                ctl_dataset, lw_var, config.lw_units
            )
        except RCEProcessingError as ctl_error:
            if not config.inspect_only:
                raise
            base_inventory["lw_detection_error"] = str(ctl_error)
            return ctl_records, "", base_inventory

        inventory = {
            **base_inventory,
            "lw_variable": lw_var,
            "ctl_lw_units": ctl_units,
        }
    return ctl_records, lw_var, inventory


def preprocess_daily_caches(config: PreprocessConfig) -> list[Path]:
    """Build all missing daily statistics files for ``config``."""

    config.validate()
    ctl_records, lw_var, inventory = _preflight(config)
    if config.inspect_only:
        inventory["lw_acre_flux_units"] = validate_lw_acre_inputs(config, ctl_records)
        inventory["lw_acre_flux_variables"] = list(LW_ACRE_FLUX_VARIABLES)
        print(json.dumps(inventory, indent=2))
        return []
    two_d_index = TimeIndexed2DFields(
        config.ctl_root,
        config.start_day,
        config.end_day,
        config.time_tolerance,
    )
    legacy = LegacyDiagnosticsCache(config.pickle_dir, config.time_tolerance)
    records_by_day = group_records_by_day(ctl_records, config.start_day, config.end_day)
    written: list[Path] = []

    for day, records in records_by_day.items():
        output_path = daily_cache_path(config.cache_dir, day)
        if output_path.exists() and not config.overwrite:
            _validate_existing_daily(output_path, day, lw_var, config)
            if config.verbose:
                print(f"Reusing {output_path}")
            written.append(output_path)
            continue

        if config.verbose:
            print(f"Processing day {day} ({len(records)} CTL times)")
        accumulator = _daily_accumulator()
        pclass_sidecar = pclass_cache_path(config.cache_dir, day)
        pclass_values: dict[int, np.ndarray] = {}
        generated_pclass = False
        pclass_source_counts = {"sidecar": 0, "legacy": 0, "generated": 0}
        x_coord: np.ndarray | None = None
        y_coord: np.ndarray | None = None
        source_map = {
            _time_key(record.time, config.time_tolerance): record.path
            for record in records
        }
        lw_units_seen: str | None = None
        rain_units_seen: str | None = None
        lw_acre_units_seen: dict[str, str] = {}

        for record in records:
            with xr.open_dataset(record.path, decode_times=False) as ctl_dataset:
                current_x, current_y = xy_coordinates(ctl_dataset)
                if x_coord is None:
                    x_coord, y_coord = current_x, current_y
                    pclass_values.update(
                        load_pclass_sidecar(
                            pclass_sidecar,
                            [item.time for item in records],
                            x_coord,
                            y_coord,
                            source_map,
                            config.time_tolerance,
                        )
                    )
                elif not np.allclose(x_coord, current_x) or not np.allclose(
                    y_coord, current_y
                ):
                    raise RCEProcessingError("Horizontal grid changes within a day")

                ctl_pressure = pressure_from_dataset(ctl_dataset)
                ctl_temperature = _squeezed_array(ctl_dataset, "TABS", 3)
                ctl_qv = mixing_ratio_from_dataset(ctl_dataset, "QV")
                ctl_lw, ctl_lw_units = _read_lw_heating(
                    ctl_dataset, lw_var, config.lw_units
                )
                if lw_units_seen is not None and lw_units_seen != ctl_lw_units:
                    raise RCEProcessingError(
                        f"CTL LW units change within day {day}: "
                        f"{lw_units_seen!r} versus {ctl_lw_units!r}"
                    )
                lw_units_seen = ctl_lw_units
                ctl_lw = lw_heating_anomaly(ctl_lw)

                shape = ctl_pressure.shape[1:]
                ctl_crh = legacy.get(
                    "ctl",
                    "crh",
                    record.time,
                    shape,
                    current_x,
                    current_y,
                    record.path,
                )
                if ctl_crh is None:
                    ctl_crh = compute_crh(ctl_qv, ctl_temperature, ctl_pressure)
                key = _time_key(record.time, config.time_tolerance)
                if key in pclass_values:
                    pclass = pclass_values[key]
                    pclass_source_counts["sidecar"] += 1
                else:
                    cached = legacy.get(
                        "ctl",
                        "pclass",
                        record.time,
                        shape,
                        current_x,
                        current_y,
                        record.path,
                    )
                    if cached is not None:
                        pclass = validate_pclass(cached, shape)
                        pclass_source_counts["legacy"] += 1
                    else:
                        qn = mixing_ratio_from_dataset(ctl_dataset, "QN")
                        qp = mixing_ratio_from_dataset(ctl_dataset, "QP")
                        pclass = generate_pclass(
                            ctl_temperature, qn, qp, ctl_pressure
                        )
                        generated_pclass = True
                        pclass_source_counts["generated"] += 1
                    pclass_values[key] = pclass

                w = align_vertical_velocity(
                    _squeezed_array(ctl_dataset, "W", 3), ctl_pressure.shape[0]
                )
                rho = density_from_state(ctl_pressure, ctl_temperature, ctl_qv)
                ctl_interpolated = interpolate_fields_to_pressure(
                    {"lw": ctl_lw, "mass_flux": rho * w}, ctl_pressure
                )
                ctl_bins = assign_crh_bins(ctl_crh)
                nbins = CRH_EDGES.size - 1
                accumulator["crh_sample_count"] += np.bincount(
                    ctl_bins[ctl_bins >= 0], minlength=nbins
                )[:nbins]

                sums, counts = binned_profile_sums(
                    ctl_interpolated["lw"], ctl_bins, nbins
                )
                accumulator["lw_heating_sum"] += sums
                accumulator["lw_heating_count"] += counts

                sums, counts = binned_profile_sums(
                    ctl_interpolated["mass_flux"], ctl_bins, nbins
                )
                accumulator["mass_flux_sum"] += sums
                accumulator["mass_flux_count"] += counts

                rain, rain_units, rain_x, rain_y = two_d_index.read_rain(
                    record.time, config.rain_var, config.rain_units
                )
                if not np.allclose(current_x, rain_x) or not np.allclose(
                    current_y, rain_y
                ):
                    raise RCEProcessingError("3-D and precipitation grids differ")
                if rain_units_seen is not None and rain_units_seen != rain_units:
                    raise RCEProcessingError(
                        f"rain units change within day {day}: "
                        f"{rain_units_seen!r} versus {rain_units!r}"
                    )
                rain_units_seen = rain_units
                sums, counts = binned_scalar_sums(rain, ctl_bins, nbins)
                accumulator["rain_sum"] += sums
                accumulator["rain_count"] += counts
                acre, acre_units, acre_x, acre_y = two_d_index.read_lw_acre(record.time)
                if not np.allclose(current_x, acre_x) or not np.allclose(
                    current_y, acre_y
                ):
                    raise RCEProcessingError("3-D and LW-ACRE flux grids differ")
                for name, units in acre_units.items():
                    previous = lw_acre_units_seen.get(name)
                    if previous is not None and previous != units:
                        raise RCEProcessingError(
                            f"{name} units change within day {day}: "
                            f"{previous!r} versus {units!r}"
                        )
                    lw_acre_units_seen[name] = units
                sums, counts = binned_scalar_sums(acre, ctl_bins, nbins)
                accumulator["lw_acre_sum"] += sums
                accumulator["lw_acre_count"] += counts
                accumulator["pclass_count"] += binned_class_counts(
                    pclass, ctl_bins, nbins
                )

        if x_coord is None or y_coord is None or rain_units_seen is None or not lw_acre_units_seen:
            raise RCEProcessingError(f"No data accumulated for day {day}")
        if generated_pclass:
            ordered = sorted(records, key=lambda record: record.time)
            write_pclass_sidecar(
                pclass_sidecar,
                [record.time for record in ordered],
                [pclass_values[_time_key(record.time, config.time_tolerance)] for record in ordered],
                x_coord,
                y_coord,
                [record.path for record in ordered],
            )

        dataset = _daily_dataset(
            accumulator,
            day,
            records,
            lw_var,
            lw_units_seen,
            rain_units_seen,
            pclass_source_counts,
            lw_acre_units_seen,
            config.rain_var,
        )
        atomic_to_netcdf(dataset, output_path, _numeric_encoding(dataset))
        written.append(output_path)
    return written


def _sum_daily_variable(datasets: Sequence[xr.Dataset], name: str) -> np.ndarray:
    return np.sum([np.asarray(dataset[name].values) for dataset in datasets], axis=0)


def aggregate_daily_cache(
    cache_dir: Path | str,
    start_day: int = 1,
    end_day: int = 21,
    minimum_bin_samples: int = 4,
) -> xr.Dataset:
    """Combine daily sufficient statistics for a half-open day range."""

    if end_day <= start_day:
        raise ValueError("end_day must be greater than start_day")
    paths = [daily_cache_path(cache_dir, day) for day in range(start_day, end_day)]
    missing = [path for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing daily caches:\n" + "\n".join(str(path) for path in missing)
        )

    datasets = [xr.load_dataset(path) for path in paths]
    try:
        reference = datasets[0]
        for dataset, path in zip(datasets, paths):
            if dataset.attrs.get("schema_version") != SCHEMA_VERSION:
                raise CacheValidationError(f"Unsupported daily schema in {path}")
            for coordinate in ("pressure_hpa", "crh_bin", "crh_edge", "class_code"):
                if not np.array_equal(dataset[coordinate], reference[coordinate]):
                    raise CacheValidationError(f"Coordinate {coordinate} differs in {path}")
            for attribute in (
                "lw_variable",
                "sampling_selection",
                "lw_anomaly_definition",
                "heating_diagnostic_label",
                "lw_acre_flux_variables",
                "lw_acre_flux_units",
                "classifier_version",
            ):
                if dataset.attrs.get(attribute) != reference.attrs.get(attribute):
                    raise CacheValidationError(f"Attribute {attribute} differs in {path}")

        lw_sum = _sum_daily_variable(datasets, "lw_heating_sum")
        lw_count = _sum_daily_variable(datasets, "lw_heating_count")
        sample_count = _sum_daily_variable(datasets, "crh_sample_count")
        mass_sum = _sum_daily_variable(datasets, "mass_flux_sum")
        mass_count = _sum_daily_variable(datasets, "mass_flux_count")
        rain_sum = _sum_daily_variable(datasets, "rain_sum")
        rain_count = _sum_daily_variable(datasets, "rain_count")
        acre_sum = _sum_daily_variable(datasets, "ctl_lw_acre_sum")
        acre_count = _sum_daily_variable(datasets, "ctl_lw_acre_count")
        class_count = _sum_daily_variable(datasets, "pclass_count")

        lw_mean = safe_mean(lw_sum, lw_count)
        mass_mean = safe_mean(mass_sum, mass_count)
        rain_mean = safe_mean(rain_sum, rain_count)
        acre_mean = safe_mean(acre_sum, acre_count)
        ctl_valid_bin = sample_count >= minimum_bin_samples
        lw_mean[:, ~ctl_valid_bin] = np.nan
        mass_mean[:, ~ctl_valid_bin] = np.nan
        rain_mean[~ctl_valid_bin] = np.nan
        acre_mean[~ctl_valid_bin] = np.nan
        class_count_for_area = class_count.astype(float)
        class_count_for_area[:, ~ctl_valid_bin] = np.nan
        class_percent = global_class_area_percent(class_count_for_area)
        output = xr.Dataset(
            data_vars={
                "lw_heating_sum": (
                    ("pressure_hpa", "crh_bin"),
                    lw_sum,
                ),
                "lw_heating_count": (
                    ("pressure_hpa", "crh_bin"),
                    lw_count,
                ),
                "crh_sample_count": (
                    ("crh_bin",), sample_count
                ),
                "lw_heating_mean": (
                    ("pressure_hpa", "crh_bin"),
                    lw_mean,
                ),
                "ctl_mass_flux": (
                    ("pressure_hpa", "crh_bin"), mass_mean
                ),
                "ctl_mass_flux_count": (
                    ("pressure_hpa", "crh_bin"), mass_count
                ),
                "ctl_precip_mm_day": (("crh_bin",), rain_mean),
                "ctl_precip_mm_hour": (("crh_bin",), rain_mean / 24.0),
                "ctl_precip_count": (("crh_bin",), rain_count),
                "ctl_lw_acre_sum": (("crh_bin",), acre_sum),
                "ctl_lw_acre_count": (("crh_bin",), acre_count),
                "ctl_lw_acre": (("crh_bin",), acre_mean),
                "ctl_pclass_count": (
                    ("class_code", "crh_bin"), class_count
                ),
                "ctl_class_area_percent": (
                    ("class_code", "crh_bin"), class_percent
                ),
                "crh_bin_center": (
                    ("crh_bin",), reference["crh_bin_center"].values
                ),
            },
            coords={
                "pressure_hpa": reference["pressure_hpa"].values,
                "crh_bin": reference["crh_bin"].values,
                "crh_edge": reference["crh_edge"].values,
                "class_code": reference["class_code"].values,
                "class_name": (
                    ("class_code",), reference["class_name"].values
                ),
            },
            attrs={
                "schema_version": SCHEMA_VERSION,
                "start_day_inclusive": int(start_day),
                "end_day_exclusive": int(end_day),
                "time_bounds": f"[{start_day}, {end_day})",
                "number_of_days": int(end_day - start_day),
                "minimum_bin_samples": int(minimum_bin_samples),
                "lw_variable": reference.attrs["lw_variable"],
                "sampling_selection": reference.attrs["sampling_selection"],
                "lw_anomaly_definition": reference.attrs["lw_anomaly_definition"],
                "heating_diagnostic_label": reference.attrs["heating_diagnostic_label"],
                "diagnostic_label": "CTL local flux-derived LW cloud-radiative effect",
                "scientific_caveat": (
                    "ctl_lw_acre is an instantaneous local all-sky minus clear-sky "
                    "flux diagnostic. lw_heating_mean is the CTL local longwave "
                    "heating anomaly conditioned on CTL CRH."
                ),
                "lw_acre_definition": reference.attrs["lw_acre_definition"],
                "lw_acre_flux_variables": reference.attrs["lw_acre_flux_variables"],
                "lw_acre_flux_units": reference.attrs["lw_acre_flux_units"],
                "class_area_definition": (
                    "100*N(class,CRH-bin)/N(all primitive classes,all CRH bins)"
                ),
                "source_daily_files": json.dumps([str(path) for path in paths]),
            },
        )
        output["lw_heating_mean"].attrs.update(
            units="K day-1",
            long_name="conditional mean CTL local longwave radiative heating anomaly",
        )
        output["ctl_mass_flux"].attrs.update(
            units="kg m-2 s-1", long_name="CTL pressure-resolved vertical mass flux"
        )
        output["ctl_precip_mm_day"].attrs["units"] = "mm day-1"
        output["ctl_precip_mm_hour"].attrs["units"] = "mm hour-1"
        output["ctl_class_area_percent"].attrs["units"] = "%"
        output["ctl_lw_acre"].attrs.update(
            units="W m-2",
            long_name="conditional mean local CTL longwave cloud-radiative effect",
            positive="cloud-induced atmospheric LW flux convergence",
        )
        output["pressure_hpa"].attrs["units"] = "hPa"
        return output
    finally:
        for dataset in datasets:
            dataset.close()


def save_aggregate_dataset(
    dataset: xr.Dataset, output_dir: Path | str | None = None
) -> Path:
    """Save an aggregated window dataset with a deterministic name."""

    start_day = int(dataset.attrs["start_day_inclusive"])
    end_day = int(dataset.attrs["end_day_exclusive"])
    directory = Path(output_dir) if output_dir is not None else Path(".")
    path = aggregate_output_path(directory, start_day, end_day)
    return atomic_to_netcdf(dataset, path, _numeric_encoding(dataset))


def _robust_symmetric_limit(values: np.ndarray, percentile: float = 99.0) -> float:
    finite = np.abs(np.asarray(values, dtype=float))
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        raise RCEProcessingError("No finite displayed values are available for plotting")
    limit = float(np.nanpercentile(finite, percentile))
    return limit if limit > 0 else 1.0


def _mass_flux_levels(values: np.ndarray) -> np.ndarray:
    limit = _robust_symmetric_limit(values, 99.0)
    # positive = np.unique(limit * np.asarray([0.10, 0.25, 0.50, 0.75]))
    positive = np.unique(limit * np.asarray([0.001, 0.01, 0.05, 0.10, 0.25, 0.50]))
    positive = positive[positive > 0]
    return np.concatenate((-positive[::-1], positive))


def plot_cross_section(
    dataset: xr.Dataset,
    output_dir: Path | str | None = None,
    show: bool = False,
    dpi: int = 400,
):
    """Create and optionally save the two-panel RCE cross-section figure."""

    import matplotlib.pyplot as plt
    from matplotlib import colors, rc
    from matplotlib.lines import Line2D
    from matplotlib.ticker import ScalarFormatter, AutoLocator
    import seaborn as sns

    font = {'family' : 'sans-serif',
            'weight' : 'normal',
            'size'   : 10}

    rc('font', **font)

    sns.set_theme(style="ticks", font_scale=1.2, rc={'xtick.bottom': True, 'ytick.left': True,
                                    "axes.spines.right": False, "axes.spines.top": False,})

    x = np.asarray(dataset["crh_bin"].values)
    pressure = np.asarray(dataset["pressure_hpa"].values)
    delta = np.asarray(dataset["lw_heating_mean"].values)
    mass_flux = np.asarray(dataset["ctl_mass_flux"].values)
    display = (
        (pressure[:, None] >= 100.0)
        & (pressure[:, None] <= 1000.0)
        & (x[None, :] >= 0.40)
        & (x[None, :] <= 0.95)
    )
    color_limit = _robust_symmetric_limit(np.where(display, delta, np.nan))
    contour_levels = _mass_flux_levels(np.where(display, mass_flux, np.nan))

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(8, 6),
        height_ratios=[0.65, 0.35],
        layout="constrained",
        dpi=200,
    )
    upper, lower = axes
    norm = colors.TwoSlopeNorm(vmin=-color_limit, vcenter=0.0, vmax=color_limit)
    image = upper.pcolormesh(
        x,
        pressure,
        delta,
        cmap="RdBu_r",
        norm=norm,
        alpha=0.8,
        shading="auto",
        zorder=2,
    )
    colorbar = fig.colorbar(
        image,
        ax=upper,
        shrink=0.75,
        pad=-0.28,
        ticks=AutoLocator(),
        # label="Estimated LW forcing [K day$^{-1}$]",
        label="K day$^{-1}$",
        extend="both",
    )
    contour = upper.contour(
        x,
        pressure,
        mass_flux,
        levels=contour_levels,
        colors="black",
        # linewidths=0.9,
        zorder=3,
    )
    if contour.levels.size:
        upper.clabel(contour, contour.levels, inline=True, fontsize=13, fmt="%.3g")
    upper.set_ylabel("Pressure [hPa]")
    upper.set_xlim(0.40, 0.95)
    upper.set_ylim(1000.0, 100.0)
    upper.set_yscale("log")
    upper.set_xticks([])
    upper.set_title(
        "CTL local LW-heating anomaly (context)\n"
        "Contours: CTL vertical mass flux $\\rho w$ [kg m$^{-2}$ s$^{-1}$]",
        fontsize=11,
    )
    upper.yaxis.set_major_formatter(ScalarFormatter())
    upper.yaxis.set_minor_formatter(ScalarFormatter())

    class_area = dataset["ctl_class_area_percent"]
    class_styles = {
        1: ("Deep", "-"),
        4: ("Strat", "--"),
        5: ("Anvil", ":"),
    }
    for code, (label, linestyle) in class_styles.items():
        lower.plot(
            x,
            class_area.sel(class_code=code),
            linestyle+'r',
            # color="red",
            # linestyle=linestyle,
            linewidth=1.6,
            label=label,
        )
    lower.set_xlim(0.40, 0.95)
    lower.set_xlabel("Column saturation fraction")
    lower.set_ylabel("Area fraction [%]")

    rain_axis = lower.twinx()
    rain_axis.plot(
        x,
        dataset["ctl_precip_mm_hour"],
        color="black",
        linestyle="-",
        linewidth=1.6,
    )
    rain_axis.set_ylabel(r"$P$ [mm hour$^{-1}$]")
    rain_axis.spines["right"].set_position(("outward", 5))

    acre_axis = lower.twinx()
    acre_axis.plot(
        x,
        dataset["ctl_lw_acre"],
        color="black",
        linestyle="--",
        linewidth=1.6,
    )
    acre_axis.set_ylabel("LW-ACRE\n[W m$^{-2}$]")
    acre_axis.spines["right"].set_position(("outward", 62))
    handles, labels = lower.get_legend_handles_labels()
    handles.extend(
        [
            Line2D([0], [0], color="black", linestyle="-", label=r"$P$"),
            Line2D([0], [0], color="black", linestyle="--", label="CTL LW-ACRE"),
        ]
    )
    labels.extend([r"$P$", "CTL LW-ACRE"])
    lower.legend(handles, labels, frameon=False, prop={'size': 12})#, fontsize=10)

    for axis in (upper, lower, rain_axis, acre_axis):
        axis.spines["top"].set_visible(False)

    upper.spines["bottom"].set_visible(False)
    # fig.text(
    #     0.5,
    #     -0.015,
    #     dataset.attrs.get("scientific_caveat", ""),
    #     ha="center",
    #     va="top",
    #     fontsize=7,
    #     wrap=True,
    # )

    # Modify axes
    sns.despine(offset=10, ax=upper, bottom=True)
    sns.despine(offset=10, ax=lower, left=False, bottom=False, right=True, top=True)
    sns.despine(offset=10,ax=rain_axis, left=True, bottom=True, right=False, top=True)
    sns.despine(offset=10,ax=acre_axis, left=True, bottom=True, right=False, top=True)
    rain_axis.spines['right'].set_position(('outward', 5))
    acre_axis.spines['right'].set_position(('outward', 57))

    saved: dict[str, Path] = {}
    if output_dir is not None:
        directory = Path(output_dir)
        directory.mkdir(parents=True, exist_ok=True)
        tag = window_tag(
            int(dataset.attrs["start_day_inclusive"]),
            int(dataset.attrs["end_day_exclusive"]),
        )
        stem = directory / f"rce_ctl_lw_anomaly_crh_cross_section_{tag}"
        for suffix in ("png", "pdf"):
            path = stem.with_suffix(f".{suffix}")
            fig.savefig(path, dpi=dpi if suffix == "png" else None, bbox_inches="tight")
            saved[suffix] = path
    if show:
        plt.show()
    return fig, {
        "upper": upper,
        "lower": lower,
        "rain": rain_axis,
        "acre": acre_axis,
        "colorbar": colorbar,
        "contour": contour,
        "saved": saved,
        "color_limit": color_limit,
        "contour_levels": contour_levels,
    }


def build_preprocess_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build daily CRH-binned statistics for the SAM RCE CTL local "
            "longwave-heating anomaly."
        )
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--ctl-root", type=Path)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--pickle-dir", type=Path)
    parser.add_argument("--start-day", type=int, default=1)
    parser.add_argument("--end-day", type=int, default=21)
    parser.add_argument(
        "--lw-var",
        help="3-D LW-only temperature-tendency variable; QRAD is rejected",
    )
    parser.add_argument(
        "--lw-units",
        help="Override missing/incorrect LW units (for example K/day or K/s)",
    )
    parser.add_argument("--rain-var", default="Prec")
    parser.add_argument("--rain-units")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--inspect",
        action="store_true",
        help="Validate inputs and print variable/time inventory without writing",
    )
    parser.add_argument("--quiet", action="store_true")
    return parser


def config_from_args(args: argparse.Namespace) -> PreprocessConfig:
    data_root = Path(args.data_root)
    ctl_root = Path(args.ctl_root) if args.ctl_root else data_root / DEFAULT_CTL_NAME
    cache_dir = (
        Path(args.cache_dir)
        if args.cache_dir
        else data_root / "pickle_out" / "binned_cross_2d_rce"
    )
    pickle_dir = Path(args.pickle_dir) if args.pickle_dir else data_root / "pickle_out"
    return PreprocessConfig(
        ctl_root=ctl_root,
        cache_dir=cache_dir,
        pickle_dir=pickle_dir,
        start_day=args.start_day,
        end_day=args.end_day,
        lw_var=args.lw_var,
        lw_units=args.lw_units,
        rain_var=args.rain_var,
        rain_units=args.rain_units,
        overwrite=args.overwrite,
        inspect_only=args.inspect,
        verbose=not args.quiet,
    )


def main_preprocess(argv: Sequence[str] | None = None) -> int:
    parser = build_preprocess_parser()
    args = parser.parse_args(argv)
    config = config_from_args(args)
    try:
        paths = preprocess_daily_caches(config)
    except (OSError, ValueError, RCEProcessingError) as exc:
        parser.error(str(exc))
    if paths and not args.quiet:
        print(f"Ready: {len(paths)} daily cache files in {config.cache_dir / 'daily'}")
    return 0


__all__ = [
    "CLASSIFIER_VERSION",
    "CLASS_CODES",
    "CRH_EDGES",
    "PRESSURE_HPA",
    "CacheValidationError",
    "FileRecord",
    "LW_ACRE_FLUX_VARIABLES",
    "PreprocessConfig",
    "RCEProcessingError",
    "aggregate_daily_cache",
    "align_vertical_velocity",
    "assign_crh_bins",
    "binned_class_counts",
    "binned_profile_sums",
    "binned_scalar_sums",
    "classify_integrated_hydrometeors",
    "compute_crh",
    "density_from_state",
    "flux_to_w_m2",
    "generate_pclass",
    "global_class_area_percent",
    "interpolate_fields_to_pressure",
    "interpolate_profile_to_pressure",
    "load_pclass_sidecar",
    "lw_acre_from_heating",
    "lw_acre_from_fluxes",
    "lw_heating_anomaly",
    "plot_cross_section",
    "preprocess_daily_caches",
    "save_aggregate_dataset",
    "select_half_open_times",
    "TimeIndexed2DFields",
    "validate_lw_acre_inputs",
    "window_tag",
    "write_pclass_sidecar",
]
