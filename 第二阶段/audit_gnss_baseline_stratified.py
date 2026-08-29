"""Create stratified, independent-GNSS baseline metrics for 2019 files.

The raw metrics are the primary result.  The optional bias-correction section
is a diagnostic only: the correction is fitted on the first chronological
30% of each file and evaluated on the remaining 70%, so it does not reuse the
evaluation residuals.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED = {"time", "pwv_mm", "pwv_gpt3"}


def metrics(error: pd.Series | np.ndarray) -> dict[str, float | int]:
    values = pd.to_numeric(pd.Series(error), errors="coerce").dropna().to_numpy(float)
    return {
        "N": int(values.size),
        "RMSE_mm": float(np.sqrt(np.mean(values**2))) if values.size else float("nan"),
        "MAE_mm": float(np.mean(np.abs(values))) if values.size else float("nan"),
        "Bias_mm": float(np.mean(values)) if values.size else float("nan"),
    }


def _read(files: list[Path]) -> pd.DataFrame:
    frames = []
    for path in files:
        frame = pd.read_csv(path)
        missing = REQUIRED - set(frame.columns)
        if missing:
            raise ValueError(f"{path.name} missing {sorted(missing)}")
        frame = frame.copy()
        frame["time"] = pd.to_datetime(frame["time"], errors="coerce")
        frame["station_id"] = path.name.removeprefix("gnss_pwv_").removesuffix("_gpt3profile.csv")
        frame["error_mm"] = pd.to_numeric(frame["pwv_gpt3"], errors="coerce") - pd.to_numeric(frame["pwv_mm"], errors="coerce")
        frames.append(frame)
    data = pd.concat(frames, ignore_index=True).dropna(subset=["time", "error_mm"])
    data["month"] = data["time"].dt.month
    return data.sort_values(["time", "station_id"], kind="stable").reset_index(drop=True)


def grouped_metrics(data: pd.DataFrame, group_column: str) -> list[dict[str, object]]:
    rows = []
    for key, group in data.groupby(group_column, sort=True, dropna=False):
        row = {group_column: int(key) if isinstance(key, (int, np.integer)) else str(key)}
        row.update(metrics(group["error_mm"]))
        rows.append(row)
    return rows


def chronological_bias_check(data: pd.DataFrame, fit_fraction: float = 0.3) -> dict[str, object]:
    corrected_errors = []
    details = []
    for station, group in data.groupby("station_id", sort=True):
        group = group.sort_values("time", kind="stable")
        split = max(1, min(len(group) - 1, int(len(group) * fit_fraction)))
        correction = float(group.iloc[:split]["error_mm"].mean())
        test_error = group.iloc[split:]["error_mm"] - correction
        corrected_errors.append(test_error)
        details.append({"station_id": station, "fit_N": split, "eval_N": len(test_error), "fit_bias_mm": correction})
    pooled = pd.concat(corrected_errors, ignore_index=True) if corrected_errors else pd.Series(dtype=float)
    return {"fit_fraction": fit_fraction, "evaluation": metrics(pooled), "per_station_fit": details}


def audit(input_dir: Path) -> dict[str, object]:
    files = sorted(input_dir.glob("gnss_pwv_*_gpt3profile.csv"))
    if not files:
        raise FileNotFoundError(f"no GNSS GPT3 files in {input_dir}")
    data = _read(files)
    return {
        "scope": "independent GNSS PWV reference, existing 2019 files",
        "file_count": len(files),
        "station_count": int(data["station_id"].nunique()),
        "time_min": data["time"].min().isoformat(),
        "time_max": data["time"].max().isoformat(),
        "pooled_raw": metrics(data["error_mm"]),
        "per_station_raw": grouped_metrics(data, "station_id"),
        "per_month_raw": grouped_metrics(data, "month"),
        "chronological_global_bias_diagnostic": chronological_bias_check(data),
        "interpretation": (
            "Raw pooled/per-station/month metrics are the primary baseline. "
            "The chronological bias diagnostic is supplementary and does not "
            "turn GPT3 into a retrained model."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.input_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
