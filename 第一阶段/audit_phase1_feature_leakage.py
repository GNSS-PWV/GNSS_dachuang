"""Audit historical phase-one scripts and validate the causal feature module."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from causal_features import build_causal_features, feature_contract


TARGETS = ("PS", "WPS", "TS", "Tm")
SCRIPT_DEFAULTS = (
    Path("PS/gru_ps_2.py"),
    Path("WPS/gru_wps_1.py"),
    Path("Ts_Tm/gru_tm_a800.py"),
    Path("Ts_Tm/gru_ts_tm_fixed.py"),
)


def _current_token(target: str) -> re.Pattern[str]:
    return re.compile(rf"(?:data|df)\[['\"]{re.escape(target)}['\"]\]")


def audit_source(path: Path) -> list[dict[str, object]]:
    text = path.read_text(encoding="utf-8", errors="replace")
    findings: list[dict[str, object]] = []
    group_targets: dict[str, str] = {}
    for line_no, line in enumerate(text.splitlines(), start=1):
        group_match = re.search(r"(\w+_group)\s*=.*\[['\"](PS|WPS|TS|Tm)['\"]\]", line)
        if group_match:
            group_targets[group_match.group(1)] = group_match.group(2)
        for target in TARGETS:
            if not _current_token(target).search(line):
                continue
            if re.search(r"diff|ratio|product|rolling|ma\d+|std\d+", line, re.I):
                findings.append({
                    "file": str(path),
                    "line": line_no,
                    "target": target,
                    "rule": "current_dynamic_value_in_derived_feature",
                    "text": line.strip(),
                    "severity": "high",
                })
        for variable, target in group_targets.items():
            if re.search(rf"\b{re.escape(variable)}\.rolling\(", line):
                findings.append({
                    "file": str(path),
                    "line": line_no,
                    "target": target,
                    "rule": "rolling_window_includes_current_row",
                    "text": line.strip(),
                    "severity": "high",
                })
    return findings


def _synthetic_frame(rows: int = 40) -> pd.DataFrame:
    time = pd.date_range("2014-01-01", periods=rows, freq="h")
    return pd.DataFrame({
        "TIME": time,
        "YEAR": 2014,
        "DOY": time.dayofyear,
        "LAT": 31.2,
        "LON": 121.5,
        "ELV": 10.0,
        "station_id": "SYN",
        "PS": np.arange(rows, dtype=float) + 1000,
        "WPS": np.arange(rows, dtype=float) * 0.1 + 10,
        "TS": np.arange(rows, dtype=float) * 0.2 + 280,
        "Tm": np.arange(rows, dtype=float) * 0.15 + 275,
    })


def validate_causal_module() -> dict[str, object]:
    frame = _synthetic_frame()
    checks: list[dict[str, object]] = []
    for target in TARGETS:
        features, columns = build_causal_features(frame, target=target)
        if target in columns:
            raise AssertionError(f"{target} appears in causal feature list")
        altered = frame.copy()
        altered.loc[20, target] += 100000.0
        altered_features, _ = build_causal_features(altered, target=target)
        same_at_current = np.allclose(
            features.loc[20, columns].to_numpy(float),
            altered_features.loc[20, columns].to_numpy(float),
            equal_nan=True,
        )
        if not same_at_current:
            raise AssertionError(f"current {target} changed its own feature row")
        checks.append({
            "target": target,
            "feature_count": len(columns),
            "current_mutation_does_not_change_current_features": True,
        })
    return {"checks": checks, "status": "pass"}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    paths = [args.root / path for path in SCRIPT_DEFAULTS if (args.root / path).exists()]
    findings = [finding for path in paths for finding in audit_source(path)]
    report = {
        "status": "pass" if validate_causal_module()["status"] == "pass" else "fail",
        "audited_scripts": [str(path) for path in paths],
        "historical_leakage_findings": findings,
        "historical_finding_count": len(findings),
        "causal_contracts": [feature_contract(target) for target in TARGETS],
        "causal_module_validation": validate_causal_module(),
        "interpretation": (
            "Historical findings document why old scores are not deployment scores. "
            "The new module is isolated and must be used by a future retraining job."
        ),
    }
    serialized = json.dumps(report, ensure_ascii=True, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized, encoding="utf-8")
    print(serialized, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
