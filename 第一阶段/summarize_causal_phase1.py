"""Summarize explicitly selected causal phase-one experiment artifacts.

The report separates temporal extrapolation from held-station extrapolation,
keeps reproducibility gaps visible, and never turns phase-one metrics into a
phase-two GNSS-PWV claim.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

TARGETS = ("PS", "WPS", "TS", "Tm")
PROTOCOL = "causal_past_only_leave_station_10pct_seed42_year_split"
SPLITS = ("test_2017", "val_2018", "val_leave_station")
METRIC_KEYS = ("N", "RMSE", "MAE", "Bias", "R2")
EXPECTED_FEATURE_COUNT = 81
EXPECTED_FEATURE_SHA256 = "efdb4ff5771d3d697e8531cc28b2ae5a8eeaa61104639e5b6bba1bbe66bfcb35"


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("JSON top level must be an object")
    return payload


def is_positive_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def is_finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def digest_feature_names(values: Any) -> str | None:
    if not isinstance(values, list) or not all(isinstance(item, str) for item in values):
        return None
    return hashlib.sha256("\n".join(values).encode("utf-8")).hexdigest()


def resolve_path(value: str | None, manifest_dir: Path) -> Path | None:
    if value is None:
        return None
    path = Path(value)
    return path if path.is_absolute() else manifest_dir / path


def artifact(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {"declared": False, "exists": False, "path": None, "bytes": None}
    exists = path.is_file()
    return {"declared": True, "exists": exists, "path": str(path), "bytes": path.stat().st_size if exists else None}


def valid_metric_block(block: Any) -> bool:
    if not isinstance(block, dict) or not is_positive_int(block.get("N")):
        return False
    return all(is_finite_number(block.get(key)) for key in ("RMSE", "MAE", "Bias", "R2"))


def valid_year_counts(value: Any, expected_years: set[str], expected_n: int) -> bool:
    if not isinstance(value, dict) or not value:
        return False
    if set(value) - expected_years:
        return False
    if not all(year in value and is_positive_int(count) for year, count in value.items()):
        return False
    return sum(value.values()) == expected_n


def validate_manifest(manifest: dict[str, Any]) -> tuple[list[dict[str, Any]], list[str]]:
    failures: list[str] = []
    runs = manifest.get("runs")
    if manifest.get("schema_version") != "phase1-summary-input/v1":
        failures.append("unexpected manifest schema_version")
    if not isinstance(runs, list):
        failures.append("runs must be an array")
        return [], failures
    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, run in enumerate(runs):
        prefix = f"runs[{index}]"
        if not isinstance(run, dict):
            failures.append(f"{prefix} must be an object")
            continue
        target = run.get("target")
        if target not in TARGETS:
            failures.append(f"{prefix}.target must be one of {TARGETS}")
            continue
        if target in seen:
            failures.append(f"duplicate target in manifest: {target}")
            continue
        seen.add(target)
        for field in ("metrics_path", "model_path", "scalers_path"):
            if field in run and run[field] is not None and not isinstance(run[field], str):
                failures.append(f"{prefix}.{field} must be a string or null")
        if "target_unit" in run and run["target_unit"] is not None and not isinstance(run["target_unit"], str):
            failures.append(f"{prefix}.target_unit must be a string or null")
        if "run_id" in run and run["run_id"] is not None and not isinstance(run["run_id"], (str, int)):
            failures.append(f"{prefix}.run_id must be a string, integer, or null")
        normalized.append(run)
    return normalized, failures


def summarize_run(run: dict[str, Any], manifest_dir: Path) -> dict[str, Any]:
    target = run["target"]
    record: dict[str, Any] = {
        "target": target,
        "run_id": str(run.get("run_id", "not_recorded")),
        "units": run.get("target_unit"),
        "unit_source": "manifest" if run.get("target_unit") else "not_recorded",
        "status": "missing",
        "artifact_completeness": "not_available",
        "warnings": [],
        "missing_fields": [],
    }
    metrics_path = resolve_path(run.get("metrics_path"), manifest_dir)
    model_path = resolve_path(run.get("model_path"), manifest_dir)
    scalers_path = resolve_path(run.get("scalers_path"), manifest_dir)
    record["artifacts"] = {"metrics": artifact(metrics_path), "model": artifact(model_path), "scalers": artifact(scalers_path)}
    if metrics_path is None or not metrics_path.is_file():
        record["warnings"].append("metrics artifact is not available")
        return record

    try:
        payload = read_json(metrics_path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        record["status"] = "invalid"
        record["warnings"].append(f"cannot read metrics JSON: {exc}")
        return record

    record.update({
        "source_metrics_path": str(metrics_path),
        "protocol": payload.get("protocol"),
        "feature_count": payload.get("feature_count"),
        "feature_fingerprint_sha256": digest_feature_names(payload.get("feature_names")),
        "scalers_fit_on": payload.get("scalers_fit_on"),
        "dataset_counts": payload.get("dataset_counts"),
        "year_counts": payload.get("year_counts", "not_recorded"),
        "time_steps": payload.get("time_steps", "not_recorded"),
        "metrics": payload.get("metrics"),
        "warning_from_training": payload.get("warning"),
        "metrics_schema_version": payload.get("schema_version", "legacy_unversioned"),
    })
    payload_unit = payload.get("target_unit")
    if isinstance(payload_unit, str) and payload_unit.strip():
        record["units"] = payload_unit
        record["unit_source"] = "metrics"

    required_top = ("protocol", "target", "feature_count", "feature_names", "dataset_counts", "scalers_fit_on", "metrics")
    record["missing_fields"] = [key for key in required_top if key not in payload]
    failures: list[str] = []
    if payload.get("target") != target:
        failures.append("manifest target disagrees with metrics target")
    if payload.get("protocol") != PROTOCOL:
        failures.append("unexpected split protocol")
    if payload.get("scalers_fit_on") != "train":
        failures.append("scalers_fit_on is not train")
    if payload.get("feature_count") != EXPECTED_FEATURE_COUNT or payload.get("feature_count") != len(payload.get("feature_names", [])):
        failures.append("feature_count does not match the registered 81-feature contract")
    if record["feature_fingerprint_sha256"] != EXPECTED_FEATURE_SHA256:
        failures.append("feature_names do not match the registered causal feature contract")

    metric_payload = payload.get("metrics")
    count_payload = payload.get("dataset_counts")
    if not isinstance(metric_payload, dict) or not isinstance(count_payload, dict):
        failures.append("metrics or dataset_counts is invalid")
    else:
        for split in SPLITS:
            block = metric_payload.get(split)
            expected_n = count_payload.get(split)
            if not valid_metric_block(block):
                failures.append(f"metric split is missing or invalid: {split}")
            elif not is_positive_int(expected_n) or block["N"] != expected_n:
                failures.append(f"dataset_counts and metric N disagree for {split}")

    if record["missing_fields"] or failures:
        record["status"] = "invalid"
        record["warnings"].extend(failures)
        return record

    record["status"] = "valid"
    year_payload = payload.get("year_counts")
    year_complete = isinstance(year_payload, dict) and valid_year_counts(year_payload.get("train"), {"2014", "2015", "2016"}, count_payload["train"]) and valid_year_counts(year_payload.get("test_2017"), {"2017"}, count_payload["test_2017"]) and valid_year_counts(year_payload.get("val_2018"), {"2018"}, count_payload["val_2018"]) and valid_year_counts(year_payload.get("val_leave_station"), {"2014", "2015", "2016", "2017", "2018"}, count_payload["val_leave_station"])
    metric_metadata_complete = isinstance(payload_unit, str) and bool(payload_unit.strip()) and is_positive_int(payload.get("time_steps"))
    all_artifacts = all(record["artifacts"][name]["exists"] for name in ("metrics", "model", "scalers"))
    if all_artifacts and year_complete and metric_metadata_complete:
        record["artifact_completeness"] = "complete"
    else:
        record["artifact_completeness"] = "legacy_incomplete"
        if not record["artifacts"]["model"]["exists"]:
            record["warnings"].append("model artifact is missing")
        if not record["artifacts"]["scalers"]["exists"]:
            record["warnings"].append("scaler artifact is missing")
        if not year_complete:
            record["warnings"].append("year_counts is missing or fails split-count validation")
        if not isinstance(payload_unit, str) or not payload_unit.strip():
            record["warnings"].append("target_unit was not recorded in metrics; manifest unit is informational only")
        if not is_positive_int(payload.get("time_steps")):
            record["warnings"].append("time_steps was not recorded in metrics")
    if isinstance(payload_unit, str) and run.get("target_unit") and payload_unit != run["target_unit"]:
        record["status"] = "invalid"
        record["warnings"].append("manifest target_unit disagrees with metrics target_unit")
    return record


def render_metrics(record: dict[str, Any], split: str) -> list[str]:
    metrics = record.get("metrics")
    block = metrics.get(split, {}) if isinstance(metrics, dict) else {}
    if not valid_metric_block(block):
        return ["—"] * 5
    return [f"{block['N']:,}", f"{block['RMSE']:.4f}", f"{block['MAE']:.4f}", f"{block['Bias']:+.4f}", f"{block['R2']:.6f}"]


def render_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Causal phase-one result summary",
        "",
        f"Generated (UTC): {summary['generated_at_utc']}",
        "",
        "## Scientific boundaries",
        "",
        "- These are phase-one PS/WPS/TS/Tm past-only forecasts, not independent phase-two GNSS-PWV retrieval results.",
        "- 2017/2018 are temporal extrapolation on non-held stations. Held-station validation is spatial/station extrapolation over 2014-2018; do not merge these scores.",
        "- N is the number of overlapping fixed-window sequences, not the number of independent weather processes or stations.",
        "",
        "## Temporal extrapolation",
        "",
        "| Target | Unit | Completeness | 2017 N | 2017 RMSE | 2017 MAE | 2017 Bias | 2017 R2 | 2018 N | 2018 RMSE | 2018 MAE | 2018 Bias | 2018 R2 |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for target in TARGETS:
        record = summary["targets"][target]
        lines.append("| " + " | ".join([target, record.get("units") or "not_recorded", record["artifact_completeness"], *render_metrics(record, "test_2017"), *render_metrics(record, "val_2018")]) + " |")
    lines += ["", "## Held-station extrapolation", "", "| Target | Unit | N | RMSE | MAE | Bias | R2 |", "|---|---|---:|---:|---:|---:|---:|"]
    for target in TARGETS:
        record = summary["targets"][target]
        lines.append("| " + " | ".join([target, record.get("units") or "not_recorded", *render_metrics(record, "val_leave_station")]) + " |")
    lines += ["", "## Artifact warnings", ""]
    for target in TARGETS:
        record = summary["targets"][target]
        details = "; ".join(record.get("warnings", [])) or "none"
        lines.append(f"- **{target}**: status `{record['status']}`, completeness `{record['artifact_completeness']}`; {details}.")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path, help="explicit run manifest JSON")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--allow-incomplete", action="store_true", help="write a partial report with exit code 0; invalid results still fail")
    args = parser.parse_args()

    manifest_path = args.manifest.resolve()
    manifest = read_json(manifest_path)
    runs, manifest_failures = validate_manifest(manifest)
    by_target = {run["target"]: run for run in runs}
    summary: dict[str, Any] = {
        "schema_version": "phase1-causal-summary/v2",
        "generated_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "common_contract": {"protocol": PROTOCOL, "expected_feature_count": EXPECTED_FEATURE_COUNT, "expected_feature_sha256": EXPECTED_FEATURE_SHA256},
        "targets": {},
        "comparability_warnings": manifest_failures.copy(),
    }
    for target in TARGETS:
        if target in by_target:
            summary["targets"][target] = summarize_run(by_target[target], manifest_path.parent)
        else:
            summary["targets"][target] = {"target": target, "units": None, "unit_source": "not_recorded", "status": "missing", "artifact_completeness": "not_available", "warnings": ["not included in manifest"], "missing_fields": []}
    records = list(summary["targets"].values())
    statuses = [record["status"] for record in records]
    if manifest_failures or "invalid" in statuses:
        summary["status"] = "invalid"
    elif not all(status == "valid" for status in statuses):
        summary["status"] = "incomplete"
    elif all(record["artifact_completeness"] == "complete" for record in records):
        summary["status"] = "complete"
    else:
        summary["status"] = "complete_metrics_with_legacy_artifacts"

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "phase1_causal_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "phase1_causal_summary.md").write_text(render_markdown(summary), encoding="utf-8")
    print(json.dumps({"status": summary["status"], "output_dir": str(args.output_dir)}, ensure_ascii=False))
    return 0 if summary["status"] == "complete" or (args.allow_incomplete and summary["status"] != "invalid") else 1


if __name__ == "__main__":
    raise SystemExit(main())