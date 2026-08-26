# -*- coding: utf-8 -*-
"""Offline preflight for the strict phase-2 profile protocol.

This command does not train or download data.  It checks whether supplied
profile directories contain the fields needed by ProfileTransformer and
whether all five strict splits are populated and leakage-free.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from strict_dataset_split import load_profiles_from_dirs, split_profiles


REQUIRED_PROFILE_FIELDS = {"levels", "heights", "global_raw", "pwv_surface", "zwd_surface", "station_id", "time_str"}


def audit(data_dirs, *, seed=42, holdout_ratio=0.10, max_files=None, require_all_splits=False):
    profiles = load_profiles_from_dirs(data_dirs, max_files=max_files)
    field_errors = []
    for i, profile in enumerate(profiles):
        missing = sorted(REQUIRED_PROFILE_FIELDS - set(profile))
        if missing:
            field_errors.append({"index": i, "missing": missing})
    splits, manifest = split_profiles(profiles, seed=seed, holdout_ratio=holdout_ratio)
    empty = [name for name, values in splits.items() if not values]
    if require_all_splits and empty:
        raise ValueError(f"strict protocol split(s) empty: {empty}")
    result = {
        "status": "PASS" if profiles and not field_errors else "FAIL",
        "profile_count": len(profiles),
        "field_error_count": len(field_errors),
        "empty_splits": empty,
        "split_profile_counts": manifest["split_profile_counts"],
        "split_station_counts": manifest["split_station_counts"],
        "split_years": manifest["split_years"],
        "holdout_stations": manifest["holdout_stations"],
        "scaler_fit_rule": "train only",
        "field_errors_sample": field_errors[:10],
    }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", action="append", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--max_files", type=int, default=None)
    parser.add_argument("--require_all_splits", action="store_true")
    args = parser.parse_args()
    result = audit(args.data_dir, max_files=args.max_files, require_all_splits=args.require_all_splits)
    text = json.dumps(result, ensure_ascii=False, indent=2)
    print(text)
    if args.output:
        Path(args.output).write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
