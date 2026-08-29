"""Validate a completed causal phase-one result directory."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


REQUIRED_METRIC_SPLITS = ("test_2017", "val_2018", "val_leave_station")


def validate(result_dir: Path) -> dict[str, object]:
    metrics_path = result_dir / "metrics.json"
    model_path = result_dir / "model.pth"
    scalers_path = result_dir / "scalers.pkl"
    missing = [str(path.name) for path in (metrics_path, model_path, scalers_path) if not path.exists()]
    result: dict[str, object] = {
        "result_dir": str(result_dir),
        "status": "fail" if missing else "pass",
        "missing_files": missing,
    }
    if missing:
        return result
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    result["target"] = payload.get("target")
    result["feature_count"] = payload.get("feature_count")
    result["scalers_fit_on"] = payload.get("scalers_fit_on")
    result["split_counts"] = payload.get("dataset_counts")
    result["metrics"] = payload.get("metrics")
    failures = []
    if payload.get("scalers_fit_on") != "train":
        failures.append("scalers_fit_on is not train")
    if payload.get("protocol") != "causal_past_only_leave_station_10pct_seed42_year_split":
        failures.append("unexpected split protocol")
    for split in REQUIRED_METRIC_SPLITS:
        item = payload.get("metrics", {}).get(split)
        if not item or item.get("status") == "empty":
            failures.append(f"missing metric split: {split}")
    result["validation_failures"] = failures
    result["status"] = "pass" if not failures else "fail"
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dir", type=Path)
    args = parser.parse_args()
    report = validate(args.result_dir)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
