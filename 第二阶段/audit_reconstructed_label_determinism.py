"""Audit whether reconstructed IGRA labels are deterministically recoverable.

The experimental IGRA conversion derives PWV and ZWD from the same vertical
temperature/pressure/vapor-pressure profile.  This script independently
recomputes the conversion relationship from the columns supplied to the model
and quantifies the residual.  A near-zero residual means the data are suitable
for pipeline testing, but cannot establish independent GNSS-PWV generalization.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

G = 9.80665
EPSILON = 0.622
K2P = 22.1
K3 = 3.739e5
RHO_W = 1000.0
RV = 461.495


def pi_from_tm(tm_k: float) -> float:
    return 1e8 / (RHO_W * RV * (K3 / tm_k + K2P))


def predicted_pwv(group: pd.DataFrame) -> tuple[float, float] | None:
    """Recompute PWV from supplied levels and supplied surface ZWD."""
    g = group.sort_values("PS", ascending=False).copy()
    for column in ("PS", "TS", "WPS", "ZWD", "PWV"):
        g[column] = pd.to_numeric(g[column], errors="coerce")
    g = g.dropna(subset=["PS", "TS", "WPS", "ZWD", "PWV"])
    g = g[(g["PS"] > 0) & (g["TS"] > 150) & (g["WPS"] >= 0)]
    if len(g) < 2:
        return None
    p = g["PS"].to_numpy(float) * 100.0
    e = g["WPS"].to_numpy(float) * 100.0
    t = g["TS"].to_numpy(float)
    trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    numerator = float(trapz(e / t, p * -1.0))
    denominator = float(trapz(e / (t * t), p * -1.0))
    if denominator <= 0:
        return None
    tm = numerator / denominator
    zwd = float(g["ZWD"].iloc[0])
    truth = float(g["PWV"].iloc[0])
    return zwd * pi_from_tm(tm), truth


def audit(data_dir: Path, max_files: int | None) -> dict[str, object]:
    files = sorted(data_dir.rglob("*_met.txt"))
    if max_files is not None:
        files = files[:max_files]
    predicted: list[float] = []
    truth: list[float] = []
    skipped = 0
    for path in files:
        frame = pd.read_csv(path)
        frame["TIME"] = pd.to_datetime(frame["TIME"], errors="coerce")
        frame = frame.dropna(subset=["TIME"])
        for _, group in frame.groupby("TIME", sort=False):
            result = predicted_pwv(group)
            if result is None:
                skipped += 1
                continue
            pred, actual = result
            predicted.append(pred)
            truth.append(actual)
    if not truth:
        raise ValueError("no valid reconstructed profiles")
    pred_a = np.asarray(predicted)
    true_a = np.asarray(truth)
    error = pred_a - true_a
    return {
        "scope": "experimental reconstructed IGRA profile determinism audit",
        "files": len(files),
        "profiles": int(true_a.size),
        "skipped": skipped,
        "rmse_mm": float(np.sqrt(np.mean(error ** 2))),
        "mae_mm": float(np.mean(np.abs(error))),
        "max_abs_error_mm": float(np.max(np.abs(error))),
        "interpretation": (
            "PWV is recoverable from profile TS/PS/WPS and supplied ZWD under "
            "the same reconstruction physics; do not treat phase-2 scores on "
            "this label set as independent observational generalization."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument("--max_files", type=int, default=10)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.data_dir, args.max_files)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
