# -*- coding: utf-8 -*-
"""Build the strict station/year manifest from yearly PWV XLSX files.

This audit operates at the station-year file level because the available
``PWV??????/YYYY/*_met_PWV_result.xlsx`` files contain ``doy, h, PWV``
(target/vertical observations), not the full phase-2 radiosonde profile fields.
It therefore creates a leakage-safe assignment manifest without pretending
that XLSX files are directly trainable by phase-2 ``data.py``.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any

import numpy as np

CORE_YEARS = set(range(2014, 2019))
SPLITS = ("train", "test_2017", "val_2018", "val_leave_station", "val_2019")


def discover_records(root: str | Path) -> list[dict[str, Any]]:
    root = Path(root)
    records: list[dict[str, Any]] = []
    pattern = re.compile(r"^(.+?)_met_PWV_result\.xlsx$", re.IGNORECASE)
    for year_dir in sorted(root.iterdir()):
        if not year_dir.is_dir() or not year_dir.name.isdigit():
            continue
        year = int(year_dir.name)
        if year < 2014 or year > 2019:
            continue
        for path in sorted(year_dir.glob("*.xlsx")):
            match = pattern.match(path.name)
            if not match:
                continue
            records.append({"station_id": match.group(1), "year": year, "source_file": str(path.resolve())})
    return records


def split_records(records: list[dict[str, Any]], seed: int = 42, ratio: float = 0.10) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    core = [r for r in records if r["year"] in CORE_YEARS]
    final = [r for r in records if r["year"] == 2019]
    stations = sorted({r["station_id"] for r in core})
    n_holdout = max(1, int(len(stations) * ratio)) if stations else 0
    holdout = set(np.random.RandomState(seed).permutation(stations)[:n_holdout])
    remaining = set(stations) - holdout
    splits = {name: [] for name in SPLITS}
    for r in core:
        if r["station_id"] in holdout:
            splits["val_leave_station"].append(r)
        elif r["year"] in (2014, 2015, 2016):
            splits["train"].append(r)
        elif r["year"] == 2017:
            splits["test_2017"].append(r)
        elif r["year"] == 2018:
            splits["val_2018"].append(r)
    splits["val_2019"] = list(final)
    for values in splits.values():
        values.sort(key=lambda r: (r["station_id"], r["year"], r["source_file"]))
    seen = set()
    for name, values in splits.items():
        for r in values:
            key = (r["station_id"], r["year"], r["source_file"])
            if key in seen:
                raise AssertionError(f"duplicate assignment: {key}")
            seen.add(key)
            if name == "train" and (r["station_id"] not in remaining or r["year"] not in {2014, 2015, 2016}): raise AssertionError(name)
            if name == "test_2017" and (r["station_id"] not in remaining or r["year"] != 2017): raise AssertionError(name)
            if name == "val_2018" and (r["station_id"] not in remaining or r["year"] != 2018): raise AssertionError(name)
            if name == "val_leave_station" and (r["station_id"] not in holdout or r["year"] not in CORE_YEARS): raise AssertionError(name)
            if name == "val_2019" and r["year"] != 2019: raise AssertionError(name)
    station_sets = {name: sorted({r["station_id"] for r in values}) for name, values in splits.items()}
    manifest = {
        "rule": "leave_station_10pct_seed42_then_year_split",
        "source_format": "yearly station XLSX, one file per station-year",
        "seed": seed, "holdout_ratio": ratio,
        "core_years": sorted(CORE_YEARS), "final_validation_years": [2019],
        "n_input_files": len(records),
        "n_core_files": len(core), "n_2019_files": len(final),
        "n_stations_core": len(stations), "n_holdout_stations": len(holdout),
        "n_remaining_stations": len(remaining),
        "holdout_stations": sorted(holdout), "remaining_stations": sorted(remaining),
        "split_file_counts": {name: len(splits[name]) for name in SPLITS},
        "split_station_counts": {name: len(station_sets[name]) for name in SPLITS},
        "split_years": {name: sorted({r["year"] for r in values}) for name, values in splits.items()},
        "split_stations": station_sets,
    }
    return splits, manifest


def write_outputs(splits: dict[str, list[dict[str, Any]]], manifest: dict[str, Any], out_dir: str | Path) -> None:
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    (out / "split_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    with (out / "file_assignments.csv").open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.writer(fh); writer.writerow(["split", "station_id", "year", "source_file"])
        for name in SPLITS:
            for r in splits[name]: writer.writerow([name, r["station_id"], r["year"], r["source_file"]])
    for name in SPLITS:
        (out / f"{name}_stations.txt").write_text("\n".join(manifest["split_stations"][name]) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", required=True, help="PWV?????? root containing 2014..2019 directories")
    ap.add_argument("--output-dir", default="result_strict_xlsx_split_20260819")
    args = ap.parse_args()
    records = discover_records(args.root)
    splits, manifest = split_records(records)
    write_outputs(splits, manifest, args.output_dir)
    print(json.dumps({"profile_file_counts": manifest["split_file_counts"], "station_counts": manifest["split_station_counts"], "holdout_stations": manifest["holdout_stations"]}, ensure_ascii=False, indent=2))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
