# -*- coding: utf-8 -*-
"""Auditable station/year split required for the phase-2 dataset.

Rules implemented here:
- retain years 2014--2018 for train/test/regular validation;
- randomly hold out floor(10% of stations), at least one, with seed 42;
- held-out stations: all 2014--2018 profiles go to ``val_leave_station``;
- remaining stations: 2014--2016 train, 2017 test, 2018 validation;
- every 2019 profile goes to independent ``val_2019``.

The station universe is defined from stations having at least one profile in
2014--2018. A station present only in 2019 is not sampled for the holdout,
but its 2019 data is still retained in ``val_2019``.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

SPLIT_NAMES = ("train", "test_2017", "val_2018", "val_leave_station", "val_2019")
CORE_YEARS = frozenset(range(2014, 2019))


def _year(profile: Mapping[str, Any]) -> int:
    value = profile.get("time_str")
    if isinstance(value, dt.datetime):
        return value.year
    if isinstance(value, dt.date):
        return value.year
    text = str(value).replace("T", " ")
    return dt.datetime.fromisoformat(text[:19]).year


def _stable_station_ids(profiles: Iterable[Mapping[str, Any]]) -> list[str]:
    return sorted({str(p["station_id"]) for p in profiles if p.get("station_id")})


def split_profiles(
    profiles: Sequence[Mapping[str, Any]],
    seed: int = 42,
    holdout_ratio: float = 0.10,
) -> tuple[dict[str, list[Mapping[str, Any]]], dict[str, Any]]:
    """Split profiles and return ``(splits, manifest)``.

    Profiles outside years 2014--2019 are excluded and counted in the
    manifest. The input sequence is never mutated.
    """
    if not 0 < holdout_ratio < 1:
        raise ValueError("holdout_ratio must be between 0 and 1")

    by_year = Counter()
    excluded_years = Counter()
    core_profiles: list[Mapping[str, Any]] = []
    final_profiles: list[Mapping[str, Any]] = []
    for profile in profiles:
        try:
            year = _year(profile)
        except Exception:
            excluded_years["unparseable"] += 1
            continue
        by_year[year] += 1
        if year in CORE_YEARS:
            core_profiles.append(profile)
        elif year == 2019:
            final_profiles.append(profile)
        else:
            excluded_years[str(year)] += 1

    stations = _stable_station_ids(core_profiles)
    n_holdout = max(1, int(len(stations) * holdout_ratio)) if stations else 0
    # Sort before permutation so the result is independent of filesystem
    # ordering and Python hash randomization.
    import numpy as np
    rng = np.random.RandomState(seed)
    shuffled = list(rng.permutation(stations)) if stations else []
    holdout_stations = set(shuffled[:n_holdout])
    remaining_stations = set(stations) - holdout_stations

    splits: dict[str, list[Mapping[str, Any]]] = {name: [] for name in SPLIT_NAMES}
    for profile in core_profiles:
        station = str(profile["station_id"])
        year = _year(profile)
        if station in holdout_stations:
            splits["val_leave_station"].append(profile)
        elif year in (2014, 2015, 2016):
            splits["train"].append(profile)
        elif year == 2017:
            splits["test_2017"].append(profile)
        elif year == 2018:
            splits["val_2018"].append(profile)
    splits["val_2019"] = list(final_profiles)

    for values in splits.values():
        values.sort(key=lambda p: (str(p.get("station_id", "")), str(p.get("time_str", ""))))

    split_station_sets = {
        name: sorted({str(p["station_id"]) for p in values if p.get("station_id")})
        for name, values in splits.items()
    }
    split_year_sets = {
        name: sorted({_year(p) for p in values}) for name, values in splits.items()
    }
    manifest: dict[str, Any] = {
        "rule": "leave_station_10pct_seed42_then_year_split",
        "seed": seed,
        "holdout_ratio": holdout_ratio,
        "core_years": sorted(CORE_YEARS),
        "final_validation_years": [2019],
        "station_universe_definition": "stations with >=1 profile in 2014-2018",
        "n_input_profiles": len(profiles),
        "n_core_profiles": len(core_profiles),
        "n_2019_profiles": len(final_profiles),
        "n_excluded_profiles": sum(excluded_years.values()),
        "input_profiles_by_year": {str(k): v for k, v in sorted(by_year.items())},
        "excluded_profiles_by_year": dict(sorted(excluded_years.items())),
        "n_stations_core": len(stations),
        "n_holdout_stations": len(holdout_stations),
        "n_remaining_stations": len(remaining_stations),
        "holdout_stations": sorted(holdout_stations),
        "remaining_stations": sorted(remaining_stations),
        "split_profile_counts": {name: len(splits[name]) for name in SPLIT_NAMES},
        "split_station_counts": {name: len(split_station_sets[name]) for name in SPLIT_NAMES},
        "split_years": split_year_sets,
        "split_stations": split_station_sets,
    }
    validate_split(splits, manifest)
    return splits, manifest


def validate_split(splits: Mapping[str, Sequence[Mapping[str, Any]]], manifest: Mapping[str, Any] | None = None) -> None:
    """Raise ``AssertionError`` if the split violates the stated rules."""
    expected = set(SPLIT_NAMES)
    if set(splits) != expected:
        raise AssertionError(f"split names mismatch: {set(splits)}")
    holdout = set(manifest["holdout_stations"]) if manifest else set()
    remaining = set(manifest["remaining_stations"]) if manifest else set()
    seen: dict[tuple[str, str], str] = {}
    for split_name, values in splits.items():
        for profile in values:
            station = str(profile["station_id"])
            year = _year(profile)
            key = (station, str(profile.get("time_str")))
            if key in seen:
                raise AssertionError(f"profile leakage/duplicate assignment: {key} in {seen[key]} and {split_name}")
            seen[key] = split_name
            if split_name == "train" and (station not in remaining or year not in {2014, 2015, 2016}):
                raise AssertionError("train contains an invalid station or year")
            if split_name == "test_2017" and (station not in remaining or year != 2017):
                raise AssertionError("test_2017 contains an invalid station or year")
            if split_name == "val_2018" and (station not in remaining or year != 2018):
                raise AssertionError("val_2018 contains an invalid station or year")
            if split_name == "val_leave_station" and (station not in holdout or year not in CORE_YEARS):
                raise AssertionError("val_leave_station contains an invalid station or year")
            if split_name == "val_2019" and year != 2019:
                raise AssertionError("val_2019 contains a non-2019 profile")
    if holdout & remaining:
        raise AssertionError("holdout and remaining stations overlap")



def _read_station_file_lightweight(filepath: str | Path) -> list[dict[str, Any]]:
    """Parse the phase-2 ``*_met.txt`` format without importing torch."""
    import numpy as np
    import pandas as pd
    columns = ["TIME", "YEAR", "DOY", "LAT", "LON", "ELV", "TS", "PS", "WPS", "ZWD", "ZHD", "ZTD", "PWV", "Tm"]
    station_id = Path(filepath).name.split("_met")[0]
    try:
        df = pd.read_csv(filepath, header=0)
    except Exception:
        return []
    if len(df.columns) < len(columns):
        return []
    # CSVs written with an index carry TIME as ``Unnamed: 0``.
    if "TIME" not in df.columns:
        first = df.columns[0]
        if str(first).startswith("Unnamed") or not str(first).strip():
            df = df.rename(columns={first: "TIME"})
    # Keep the expected fields even if an input file contains extra columns.
    if not set(columns).issubset(df.columns):
        return []
    df = df[columns].copy()
    for col in ["ELV", "TS", "PS", "WPS", "ZWD", "PWV", "Tm", "LAT", "LON", "DOY", "YEAR"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["ELV", "TS", "PS", "WPS", "ZWD", "PWV", "LAT", "LON", "TIME"])
    df["TIME"] = pd.to_datetime(df["TIME"], errors="coerce")
    df = df.dropna(subset=["TIME"])
    profiles = []
    for timestamp, group in df.groupby("TIME", sort=True):
        group = group.sort_values("ELV").reset_index(drop=True)
        if len(group) < 2:
            continue
        surface = group.iloc[0]
        zwd = float(surface["ZWD"]); pwv = float(surface["PWV"])
        if zwd < 1.0 or pwv < 0.1:
            continue
        profiles.append({
            "levels": group[["ELV", "TS", "PS", "WPS"]].values.astype(np.float32),
            "heights": group["ELV"].values.astype(np.float32),
            "global_raw": {"zwd_surface": zwd, "lat": float(group["LAT"].iloc[0]), "lon": float(group["LON"].iloc[0]), "doy": float(group["DOY"].iloc[0]), "hour": float(timestamp.hour)},
            "pwv_surface": pwv, "zwd_surface": zwd, "tm_surface": float(group["Tm"].iloc[0]),
            "elv_surface": float(surface["ELV"]), "station_id": station_id, "time_str": str(timestamp),
        })
    return profiles

def load_profiles_from_dirs(data_dirs: Sequence[str | Path], max_files: int | None = None) -> list[dict[str, Any]]:
    """Load ``*_met.txt`` profiles using the existing phase-2 parser."""
    data_py = Path(__file__).resolve().with_name("data.py")
    spec = importlib.util.spec_from_file_location("phase2_data_for_split", data_py)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import {data_py}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
        read_station_file = module.read_station_file
    except ModuleNotFoundError as exc:
        if exc.name != "torch":
            raise
        # The split audit only needs parsing; torch is not required here.
        read_station_file = _read_station_file_lightweight
    profiles: list[dict[str, Any]] = []
    files: list[Path] = []
    for directory in data_dirs:
        files.extend(sorted(Path(directory).rglob("*_met.txt")))
    files = sorted(set(files))
    if max_files is not None:
        files = files[:max_files]
    for filepath in files:
        profiles.extend(read_station_file(str(filepath)))
    return profiles


def write_outputs(splits: Mapping[str, Sequence[Mapping[str, Any]]], manifest: Mapping[str, Any], output_dir: str | Path) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "split_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    with (out / "profile_assignments.csv").open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.writer(fh)
        writer.writerow(["split", "station_id", "time", "year"])
        for split_name in SPLIT_NAMES:
            for profile in splits[split_name]:
                writer.writerow([split_name, profile["station_id"], profile["time_str"], _year(profile)])
    for split_name in SPLIT_NAMES:
        (out / f"{split_name}_stations.txt").write_text(
            "\n".join(manifest["split_stations"][split_name]) + ("\n" if manifest["split_stations"][split_name] else ""),
            encoding="utf-8",
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", action="append", required=True, help="directory containing *_met.txt; repeatable")
    parser.add_argument("--output-dir", default="result_strict_split_20260819")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--holdout-ratio", type=float, default=0.10)
    parser.add_argument("--max-files", type=int, default=None)
    args = parser.parse_args()
    profiles = load_profiles_from_dirs(args.data_dir, max_files=args.max_files)
    splits, manifest = split_profiles(profiles, seed=args.seed, holdout_ratio=args.holdout_ratio)
    write_outputs(splits, manifest, args.output_dir)
    print(json.dumps({
        "output_dir": str(Path(args.output_dir).resolve()),
        "profile_counts": manifest["split_profile_counts"],
        "station_counts": manifest["split_station_counts"],
        "holdout_stations": manifest["holdout_stations"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
