"""Prepare auditable 2019 IGS ZTD -> ERA5 ZHD -> ZWD samples.

This is deliberately a *preparation* utility, not a phase-two training
launcher.  It joins the supplied 5-minute IGS ZTD records with regional,
hourly ERA5 single-level files, applies a documented nearest-grid and
station-height correction, and writes a row-level provenance table.

The supplied ERA5 ``tcwv`` is retained only as ``era5_tcwv_reference_mm``.
It is an ERA5 model product and must not be represented as an independent
PWV label or passed to the independent phase-two contract without an agreed
scientific protocol.

The files are NetCDF4/HDF5.  Reading via h5py avoids making this preparation
step depend on a particular xarray NetCDF backend on Windows.
"""
from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


G0 = 9.80665
LAPSE_RATE_K_PER_M = 0.0065
RD_J_PER_KG_K = 287.05


@dataclass(frozen=True)
class RegionSpec:
    name: str
    gnss_dir: str
    era5_dir: str
    hourly_prefix: str
    geopotential_name: str


REGIONS = (
    RegionSpec("australia", "AUSTRALIA", "AUSTRALIA", "ERA5_hourly_australia_", "ERA5_geopotential_australia.nc"),
    RegionSpec("usa_conus", "USA_CONUS", "USA_CONUS", "ERA5_hourly_us_conus_", "ERA5_geopotential_us_conus.nc"),
)


def nearest_indices(values: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """Return nearest indices for monotonic ascending or descending values."""
    values = np.asarray(values, dtype=float)
    targets = np.asarray(targets, dtype=float)
    if values.size < 2:
        return np.zeros(targets.size, dtype=int)
    if values[0] > values[-1]:
        return nearest_indices(values[::-1], targets) * -1 + (values.size - 1)
    right = np.searchsorted(values, targets, side="left")
    right = np.clip(right, 0, values.size - 1)
    left = np.clip(right - 1, 0, values.size - 1)
    return np.where(np.abs(values[right] - targets) < np.abs(values[left] - targets), right, left)


def decode_units_time(values: np.ndarray) -> pd.DatetimeIndex:
    return pd.to_datetime(np.asarray(values, dtype="int64"), unit="s", utc=True)


def verify_required_keys(handle: h5py.File, keys: set[str], path: Path) -> None:
    missing = sorted(keys - set(handle.keys()))
    if missing:
        raise ValueError(f"{path.name} is missing datasets: {', '.join(missing)}")


def gather_time_grid(dataset: h5py.Dataset, ti: np.ndarray, yi: np.ndarray, xi: np.ndarray) -> np.ndarray:
    """Gather one value per row without unsupported h5py fancy indexing.

    h5py requires a sorted index for advanced indexing and cannot combine the
    independently ordered time, latitude and longitude vectors used here.
    ERA5 has only 24 hourly grids per daily input, so loading one 2-D grid per
    occupied hour is both deterministic and modest in memory.
    """
    result = np.empty(ti.size, dtype=float)
    for hour_index in np.unique(ti):
        rows = np.flatnonzero(ti == hour_index)
        grid = np.asarray(dataset[int(hour_index)], dtype=float)
        result[rows] = grid[yi[rows], xi[rows]]
    return result


class Era5Region:
    def __init__(self, spec: RegionSpec, root: Path) -> None:
        self.spec = spec
        self.root = root
        self.era5_dir = root / "ERA5" / spec.era5_dir
        self._geopotential: np.ndarray | None = None
        self._lat: np.ndarray | None = None
        self._lon: np.ndarray | None = None

    def load_static(self) -> None:
        path = self.era5_dir / self.spec.geopotential_name
        with h5py.File(path, "r") as handle:
            verify_required_keys(handle, {"latitude", "longitude", "z"}, path)
            self._lat = handle["latitude"][:].astype(float)
            self._lon = handle["longitude"][:].astype(float)
            z = handle["z"][:]
        self._geopotential = np.asarray(z, dtype=float).squeeze() / G0
        if self._geopotential.shape != (self._lat.size, self._lon.size):
            raise ValueError(f"unexpected geopotential grid shape in {path}")

    def locate_hourly(self, day: pd.Timestamp) -> Path:
        date = day.strftime("%Y%m%d")
        candidates = []
        for path in sorted(self.era5_dir.glob(f"{self.spec.hourly_prefix}*.nc")):
            match = re.search(r"_(\d{8})_(\d{8})\.nc$", path.name)
            if match and match.group(1) <= date <= match.group(2):
                candidates.append(path)
        if len(candidates) != 1:
            raise FileNotFoundError(f"expected one ERA5 window containing {self.spec.name} {date}, found {len(candidates)}")
        return candidates[0]

    def match(self, observations: pd.DataFrame, day: pd.Timestamp) -> pd.DataFrame:
        if self._lat is None or self._lon is None or self._geopotential is None:
            self.load_static()
        path = self.locate_hourly(day)
        assert self._lat is not None and self._lon is not None and self._geopotential is not None
        with h5py.File(path, "r") as handle:
            verify_required_keys(handle, {"valid_time", "latitude", "longitude", "sp", "t2m", "tcwv"}, path)
            valid_time = decode_units_time(handle["valid_time"][:])
            lat = handle["latitude"][:].astype(float)
            lon = handle["longitude"][:].astype(float)
            if not (np.array_equal(lat, self._lat) and np.array_equal(lon, self._lon)):
                raise ValueError(f"hourly/static grid mismatch for {self.spec.name}")
            times = pd.to_datetime(observations["datetime_utc"], utc=True)
            seconds = np.abs((times.values[:, None] - valid_time.values[None, :]).astype("timedelta64[s]").astype(np.int64))
            ti = seconds.argmin(axis=1)
            lat_q = observations["latitude"].to_numpy(float)
            lon_q = observations["longitude"].to_numpy(float)
            if lon.max() > 180.0:
                lon_q = np.mod(lon_q, 360.0)
            yi = nearest_indices(lat, lat_q)
            xi = nearest_indices(lon, lon_q)
            sp_grid = gather_time_grid(handle["sp"], ti, yi, xi)
            t2m_grid = gather_time_grid(handle["t2m"], ti, yi, xi)
            tcwv = gather_time_grid(handle["tcwv"], ti, yi, xi)

        result = observations.copy()
        result["era5_file"] = path.name
        result["era5_valid_time_utc"] = valid_time[ti].astype(str)
        result["era5_time_offset_seconds"] = seconds[np.arange(len(result)), ti]
        result["era5_grid_latitude"] = lat[yi]
        result["era5_grid_longitude"] = lon[xi]
        result["era5_grid_height_m"] = self._geopotential[yi, xi]
        result["era5_surface_pressure_grid_pa"] = sp_grid
        result["era5_t2m_grid_k"] = t2m_grid
        result["era5_tcwv_reference_mm"] = tcwv
        return result


def height_correct_and_qc(frame: pd.DataFrame) -> pd.DataFrame:
    """Apply hydrostatic height correction and Saastamoinen ZHD.

    IGS provides ellipsoidal height while ERA5 geopotential height is close to
    orthometric height.  The discrepancy is kept explicit in the output; this
    is a provisional, documented model choice pending a project-wide geoid
    convention.
    """
    out = frame.copy()
    dh = out["ellipsoidal_height_m"].to_numpy(float) - out["era5_grid_height_m"].to_numpy(float)
    t_grid = out["era5_t2m_grid_k"].to_numpy(float)
    t_station = t_grid - LAPSE_RATE_K_PER_M * dh
    exponent = G0 / (LAPSE_RATE_K_PER_M * RD_J_PER_KG_K)
    pressure_station = out["era5_surface_pressure_grid_pa"].to_numpy(float) * np.power(t_station / t_grid, exponent)
    lat_rad = np.deg2rad(out["latitude"].to_numpy(float))
    height_km = out["ellipsoidal_height_m"].to_numpy(float) / 1000.0
    denominator = 1.0 - 0.00266 * np.cos(2.0 * lat_rad) - 0.00028 * height_km
    zhd = 2.2768 * (pressure_station / 100.0) / denominator
    zwd = out["ztd_mm"].to_numpy(float) - zhd
    out["height_reference"] = "station_ellipsoidal_minus_era5_geopotential_height_provisional"
    out["station_minus_era5_height_m"] = dh
    out["era5_t2m_height_corrected_k"] = t_station
    out["era5_surface_pressure_height_corrected_pa"] = pressure_station
    out["zhd_saastamoinen_mm"] = zhd
    out["zwd_gnss_minus_zhd_mm"] = zwd
    checks = {
        "ztd_range": out["ztd_mm"].between(1500.0, 3500.0),
        "ztd_sigma": out["ztd_sigma_mm"].between(0.0, 20.0),
        "era5_time": out["era5_time_offset_seconds"].le(1800),
        "zhd_range": out["zhd_saastamoinen_mm"].between(1500.0, 3200.0),
        "zwd_range": out["zwd_gnss_minus_zhd_mm"].between(0.0, 700.0),
        "tcwv_range": out["era5_tcwv_reference_mm"].between(0.0, 150.0),
    }
    out["qc_failure_reasons"] = ";".join([])
    failures = pd.Series("", index=out.index, dtype=object)
    for name, passed in checks.items():
        failures = failures.mask(~passed, failures.where(failures.eq(""), failures + ";") + name)
    out["qc_failure_reasons"] = failures
    out["qc_pass"] = failures.eq("")
    return out


def process_region(spec: RegionSpec, root: Path, row_limit: int | None) -> pd.DataFrame:
    gnss_dir = root / "GNSS" / spec.gnss_dir / "daily_csv"
    files = sorted(gnss_dir.glob("IGS_ZTD_*.csv"))
    if not files:
        raise FileNotFoundError(f"no daily GNSS CSV files under {gnss_dir}")
    era5 = Era5Region(spec, root)
    frames: list[pd.DataFrame] = []
    consumed = 0
    for path in files:
        raw = pd.read_csv(path)
        required = {"datetime_utc", "station_code", "latitude", "longitude", "ellipsoidal_height_m", "ztd_mm", "ztd_sigma_mm"}
        missing = sorted(required - set(raw.columns))
        if missing:
            raise ValueError(f"{path.name} missing GNSS columns: {', '.join(missing)}")
        if row_limit is not None:
            remaining = row_limit - consumed
            if remaining <= 0:
                break
            raw = raw.head(remaining).copy()
        if raw.empty:
            continue
        day = pd.to_datetime(raw["datetime_utc"].iloc[0], utc=True).normalize()
        matched = era5.match(raw, day)
        matched["region"] = spec.name
        matched["gnss_daily_file"] = path.name
        matched["gnss_source"] = "IGS_TRO_SINEX_daily_csv"
        matched["zwd_derivation"] = "ztd_minus_zhd_saastamoinen_era5_sp_height_corrected"
        frames.append(height_correct_and_qc(matched))
        consumed += len(raw)
    if not frames:
        raise RuntimeError(f"no rows prepared for {spec.name}")
    return pd.concat(frames, ignore_index=True)


def write_outputs(frame: pd.DataFrame, output_dir: Path, args: argparse.Namespace) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    prepared_path = output_dir / "gnss_era5_zwd_2019_us_australia.csv.gz"
    frame.to_csv(prepared_path, index=False, compression="gzip")
    station_summary = (frame.groupby(["region", "station_code"], as_index=False)
                       .agg(rows=("station_code", "size"), qc_pass_rows=("qc_pass", "sum"),
                            ztd_min_mm=("ztd_mm", "min"), ztd_max_mm=("ztd_mm", "max"),
                            zwd_median_mm=("zwd_gnss_minus_zhd_mm", "median"),
                            tcwv_median_mm=("era5_tcwv_reference_mm", "median")))
    station_summary.to_csv(output_dir / "station_summary.csv", index=False)
    reasons = frame.loc[~frame["qc_pass"], "qc_failure_reasons"].str.split(";").explode().value_counts().sort_index().to_dict()
    report = {
        "schema_version": "gnss-era5-zwd-preparation/v1",
        "purpose": "2019 independent-input preparation only; ERA5 tcwv is a reference, not a formal PWV label",
        "rows_total": int(len(frame)),
        "rows_qc_pass": int(frame["qc_pass"].sum()),
        "station_count": int(frame["station_code"].nunique()),
        "by_region": frame.groupby("region").agg(rows=("region", "size"), stations=("station_code", "nunique"), qc_pass=("qc_pass", "sum")).reset_index().to_dict("records"),
        "time_range_utc": [str(frame["datetime_utc"].min()), str(frame["datetime_utc"].max())],
        "qc_failure_counts": {str(k): int(v) for k, v in reasons.items()},
        "matching": {"temporal": "nearest ERA5 hourly value, must be within 1800 seconds", "spatial": "nearest 0.25 degree ERA5 grid point"},
        "zhd": {"formula": "Saastamoinen, 2.2768 * p_hPa / (1 - 0.00266*cos(2*lat) - 0.00028*h_km)", "pressure": "ERA5 sp vertically corrected from geopotential grid height to IGS ellipsoidal station height", "height_caveat": "ellipsoidal-vs-geopotential reference mismatch remains explicit and requires future geoid convention confirmation"},
        "scientific_boundary": "Do not use era5_tcwv_reference_mm as an independent phase-two label or submit this output to GPU training until a separate PWV reference and causal phase-one profiles are available.",
        "command": {key: str(value) for key, value in vars(args).items()},
    }
    (output_dir / "preparation_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("../2019_US_Australia_IGS_ERA5"))
    parser.add_argument("--out-dir", type=Path, default=Path("independent_2019_us_australia_era5_v1"))
    parser.add_argument("--row-limit", type=int, help="process at most this many rows per region; use for smoke tests")
    args = parser.parse_args()
    root = args.data_root.resolve()
    if not root.is_dir():
        raise SystemExit(f"data root does not exist: {root}")
    if args.row_limit is not None and args.row_limit <= 0:
        raise SystemExit("--row-limit must be positive")
    frames = [process_region(spec, root, args.row_limit) for spec in REGIONS]
    frame = pd.concat(frames, ignore_index=True).sort_values(["region", "datetime_utc", "station_code"], kind="stable")
    report = write_outputs(frame, args.out_dir.resolve(), args)
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
