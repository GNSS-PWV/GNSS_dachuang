"""Gate independent phase-two contract data before any GPU training.

The historical profile files couple profile, ZWD and PWV from the same
radiosonde reconstruction.  This tool accepts only an explicit sample-level
contract that demonstrates independent GNSS ZWD, past-only phase-one profiles
and independently sourced PWV labels.  It writes an audit report and exits
nonzero when any hard rule is violated.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


SPLITS = {"train", "test_2017", "val_2018", "val_leave_station", "val_2019"}
CORE_YEARS = {2014, 2015, 2016, 2017, 2018}
REQUIRED_COLUMNS = {
    "sample_id", "station_id", "analysis_time_utc", "split",
    "ztd_gnss_mm", "zhd_model_mm", "zwd_gnss_mm", "zwd_derivation",
    "gnss_source_id", "gnss_epoch_id", "zhd_source_id", "gnss_available_at_utc",
    "profile_path", "profile_sha256", "profile_source", "profile_issue_time_utc",
    "profile_max_observation_time_utc", "p1_model_id", "p1_model_sha256",
    "pwv_label_mm", "label_time_utc", "label_source", "label_source_id",
    "label_epoch_id", "label_available_at_utc", "input_lineage_root",
    "label_lineage_root", "match_tolerance_seconds",
}
PROFILE_REQUIRED_COLUMNS = {"ELV", "TS", "PS", "WPS"}
PROFILE_FORBIDDEN_COLUMNS = {"PWV", "ZWD", "ZHD", "ZTD", "TM", "PI"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_contract(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    raise ValueError("contract must be a CSV or Parquet file")


def parse_utc(value: object) -> pd.Timestamp | None:
    if not isinstance(value, str) or not value.strip():
        return None
    parsed = pd.to_datetime(value, utc=True, errors="coerce")
    return None if pd.isna(parsed) else parsed


def is_nonempty(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def load_split_manifest(path: Path) -> tuple[set[str], set[str]]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    held = {str(value) for value in payload.get("holdout_stations", [])}
    remaining = {str(value) for value in payload.get("remaining_stations", [])}
    if not held or not remaining or held & remaining:
        raise ValueError("split manifest must contain disjoint nonempty holdout_stations and remaining_stations")
    return held, remaining


def profile_path_for(value: str, contract_dir: Path, profile_root: Path | None) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (profile_root / path if profile_root else contract_dir / path).resolve()


def validate_profile(path: Path, expected_hash: str, cache: dict[tuple[Path, str], list[str]]) -> list[str]:
    cache_key = (path, str(expected_hash).lower())
    if cache_key in cache:
        return cache[cache_key]
    failures: list[str] = []
    if not path.is_file():
        failures.append("profile_missing")
    elif path.suffix.lower() != ".csv":
        failures.append("profile_not_csv")
    else:
        try:
            header = set(pd.read_csv(path, nrows=0).columns.str.upper())
        except (OSError, ValueError, pd.errors.ParserError):
            failures.append("profile_unreadable")
        else:
            if not PROFILE_REQUIRED_COLUMNS.issubset(header):
                failures.append("profile_missing_level_columns")
            if header & PROFILE_FORBIDDEN_COLUMNS:
                failures.append("profile_contains_forbidden_target_columns")
            if not failures and sha256_file(path).lower() != str(expected_hash).lower():
                failures.append("profile_sha256_mismatch")
    cache[cache_key] = failures
    return failures


def validate_row(row: pd.Series, contract_dir: Path, profile_root: Path | None,
                 held: set[str], remaining: set[str], profile_cache: dict[tuple[Path, str], list[str]],
                 zwd_tolerance_mm: float) -> list[str]:
    failures: list[str] = []
    for column in REQUIRED_COLUMNS:
        value = row[column]
        if pd.isna(value) or (isinstance(value, str) and not value.strip()):
            failures.append(f"missing_{column}")
    if failures:
        return failures

    split = str(row["split"])
    station = str(row["station_id"])
    analysis = parse_utc(row["analysis_time_utc"])
    gnss_available = parse_utc(row["gnss_available_at_utc"])
    profile_issue = parse_utc(row["profile_issue_time_utc"])
    profile_max_obs = parse_utc(row["profile_max_observation_time_utc"])
    label_time = parse_utc(row["label_time_utc"])
    label_available = parse_utc(row["label_available_at_utc"])
    if split not in SPLITS:
        failures.append("invalid_split")
    if not all(value is not None for value in (analysis, gnss_available, profile_issue, profile_max_obs, label_time, label_available)):
        failures.append("invalid_utc_timestamp")
        return failures

    numeric = {}
    for column in ("ztd_gnss_mm", "zhd_model_mm", "zwd_gnss_mm", "pwv_label_mm", "match_tolerance_seconds"):
        value = pd.to_numeric(row[column], errors="coerce")
        if not np.isfinite(value):
            failures.append(f"invalid_{column}")
        else:
            numeric[column] = float(value)
    if failures:
        return failures
    if not (1500.0 <= numeric["ztd_gnss_mm"] <= 3500.0 and 1500.0 <= numeric["zhd_model_mm"] <= 3200.0 and 0.0 < numeric["zwd_gnss_mm"] <= 700.0 and 0.0 < numeric["pwv_label_mm"] <= 150.0):
        failures.append("out_of_physical_range")
    if abs(numeric["zwd_gnss_mm"] - (numeric["ztd_gnss_mm"] - numeric["zhd_model_mm"])) > zwd_tolerance_mm:
        failures.append("zwd_not_ztd_minus_zhd")
    if str(row["zwd_derivation"]) != "ztd_minus_zhd":
        failures.append("invalid_zwd_derivation")
    if str(row["profile_source"]) != "phase1_causal":
        failures.append("profile_not_phase1_causal")
    if str(row["input_lineage_root"]) == str(row["label_lineage_root"]):
        failures.append("input_and_label_lineage_overlap")
    if str(row["gnss_source_id"]) == str(row["label_source_id"]):
        failures.append("gnss_and_label_source_overlap")
    if gnss_available > analysis:
        failures.append("gnss_not_available_at_analysis_time")
    if profile_issue > analysis or profile_max_obs >= analysis:
        failures.append("profile_not_strictly_past_only")
    if abs((label_time - analysis).total_seconds()) > numeric["match_tolerance_seconds"]:
        failures.append("label_outside_match_tolerance")

    year = analysis.year
    if split == "train" and (station not in remaining or year not in {2014, 2015, 2016}):
        failures.append("train_split_violation")
    if split == "test_2017" and (station not in remaining or year != 2017):
        failures.append("test_2017_split_violation")
    if split == "val_2018" and (station not in remaining or year != 2018):
        failures.append("val_2018_split_violation")
    if split == "val_leave_station" and (station not in held or year not in CORE_YEARS):
        failures.append("leave_station_split_violation")
    if split == "val_2019" and year != 2019:
        failures.append("val_2019_split_violation")

    if all(is_nonempty(row[column]) for column in ("profile_path", "profile_sha256")):
        failures.extend(validate_profile(profile_path_for(str(row["profile_path"]), contract_dir, profile_root), str(row["profile_sha256"]), profile_cache))
    return failures


def audit_contract(contract: pd.DataFrame, contract_dir: Path, profile_root: Path | None,
                   held: set[str], remaining: set[str], zwd_tolerance_mm: float) -> tuple[dict, pd.DataFrame]:
    missing_columns = sorted(REQUIRED_COLUMNS - set(contract.columns))
    if missing_columns:
        raise ValueError("contract is missing required columns: " + ", ".join(missing_columns))
    profile_cache: dict[tuple[Path, str], list[str]] = {}
    failures_by_row: list[dict[str, object]] = []
    sample_ids: dict[str, int] = {}
    epoch_splits: dict[tuple[str, str], set[str]] = defaultdict(set)
    for index, row in contract.iterrows():
        errors = validate_row(row, contract_dir, profile_root, held, remaining, profile_cache, zwd_tolerance_mm)
        sample_id = str(row["sample_id"])
        if sample_id in sample_ids:
            errors.append("duplicate_sample_id")
        sample_ids[sample_id] = index
        for field in ("gnss_epoch_id", "label_epoch_id"):
            epoch_splits[(field, str(row[field]))].add(str(row["split"]))
        if errors:
            failures_by_row.append({"row": int(index) + 2, "sample_id": sample_id, "station_id": str(row["station_id"]), "split": str(row["split"]), "errors": ";".join(sorted(set(errors)))})
    cross_split_epochs = {key for key, splits in epoch_splits.items() if len(splits) > 1}
    if cross_split_epochs:
        for index, row in contract.iterrows():
            errors = [f"cross_split_{field}" for field in ("gnss_epoch_id", "label_epoch_id") if (field, str(row[field])) in cross_split_epochs]
            if errors:
                failures_by_row.append({"row": int(index) + 2, "sample_id": str(row["sample_id"]), "station_id": str(row["station_id"]), "split": str(row["split"]), "errors": ";".join(errors)})
    errors = pd.DataFrame(failures_by_row).drop_duplicates() if failures_by_row else pd.DataFrame(columns=["row", "sample_id", "station_id", "split", "errors"])
    report = {
        "schema_version": "phase2-independent-preflight/v1",
        "status": "pass" if errors.empty else "fail",
        "n_contract_rows": int(len(contract)),
        "n_rejected_rows": int(errors["sample_id"].nunique()) if not errors.empty else 0,
        "n_accepted_rows": int(len(contract) - (errors["sample_id"].nunique() if not errors.empty else 0)),
        "failure_counts": dict(Counter(error for value in errors.get("errors", pd.Series(dtype=str)) for error in str(value).split(";") if error)),
        "split_counts": {str(key): int(value) for key, value in contract["split"].value_counts().sort_index().items()},
        "year_counts": {str(key): int(value) for key, value in pd.to_datetime(contract["analysis_time_utc"], utc=True, errors="coerce").dt.year.value_counts().sort_index().items()},
        "station_count": int(contract["station_id"].astype(str).nunique()),
        "contract_sha256": None,
    }
    return report, errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--profile-root", type=Path)
    parser.add_argument("--zwd-tolerance-mm", type=float, default=0.01)
    args = parser.parse_args()
    if args.zwd_tolerance_mm <= 0:
        raise SystemExit("--zwd-tolerance-mm must be positive")
    contract_path = args.contract.resolve()
    contract = read_contract(contract_path)
    held, remaining = load_split_manifest(args.split_manifest.resolve())
    report, errors = audit_contract(contract, contract_path.parent, args.profile_root.resolve() if args.profile_root else None, held, remaining, args.zwd_tolerance_mm)
    report["contract_path"] = str(contract_path)
    report["contract_sha256"] = sha256_file(contract_path)
    report["split_manifest_path"] = str(args.split_manifest.resolve())
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "preflight_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    errors.to_csv(args.out_dir / "preflight_rejections.csv", index=False)
    print(json.dumps(report, ensure_ascii=False))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
