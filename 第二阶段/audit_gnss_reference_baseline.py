"""Audit GPT3 against independent GNSS PWV reference files."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def metrics(error: pd.Series) -> dict[str, float | int]:
    values = pd.to_numeric(error, errors="coerce").dropna().to_numpy(float)
    return {"N": int(values.size), "RMSE_mm": float(np.sqrt(np.mean(values**2))),
            "MAE_mm": float(np.mean(np.abs(values))), "Bias_mm": float(np.mean(values))}


def audit(input_dir: Path) -> dict[str, object]:
    files = sorted(input_dir.glob("*_gpt3profile.csv"))
    if not files:
        raise FileNotFoundError(f"no *_gpt3profile.csv in {input_dir}")
    frames = []
    per_file = []
    for path in files:
        frame = pd.read_csv(path)
        missing = {"pwv_mm", "pwv_gpt3"} - set(frame.columns)
        if missing:
            raise ValueError(f"{path.name} missing {sorted(missing)}")
        frames.append(frame)
        row = metrics(frame["pwv_gpt3"] - frame["pwv_mm"])
        row["source_file"] = path.name
        per_file.append(row)
    data = pd.concat(frames, ignore_index=True)
    return {"scope": "independent GNSS reference, existing 2019 local files",
            "file_count": len(files), "station_count": len(files),
            "pooled_gpt3": metrics(data["pwv_gpt3"] - data["pwv_mm"]),
            "file_metrics": per_file,
            "interpretation": "GNSS-reference baseline; do not compare directly with true-ZWD oracle scores."}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.input_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
