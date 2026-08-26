# -*- coding: utf-8 -*-
"""Aggregate and audit Phase-1-to-Phase-2 deployment CSV results.

The script computes every regime on the same valid rows within each requested
population.  When several yearly CSVs are supplied, the total metrics are
micro-averages over the concatenated samples (never an arithmetic mean of
annual RMSE values).  It also writes annual and station-macro summaries.
"""
import argparse
import glob
import json
import os
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

REGIMES = ["real", "real_surf_p1", "clim", "clim_surf_p1", "clim_adj_p1", "gpt3"]
COLS = {regime: f"pwv_{regime}" for regime in REGIMES}
REQUIRED_COLUMNS = {"station", "pwv_true", *COLS.values()}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv", required=True, nargs="+",
        help="One or more deployment CSV paths. Shell globs are accepted.",
    )
    parser.add_argument("--test_stations", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--year", default="auto",
        help="Output label retained for compatibility; default derives it from input years.",
    )
    parser.add_argument(
        "--audit", nargs="*", default=[],
        help="Optional p1_match_audit_<year>.json files or glob patterns to summarize.",
    )
    parser.add_argument(
        "--no-plot", action="store_true", help="Skip PNG comparison plots."
    )
    return parser.parse_args()


def expand_paths(items, kind):
    paths = []
    for item in items:
        hits = sorted(glob.glob(item))
        if hits:
            paths.extend(hits)
        elif os.path.isfile(item):
            paths.append(item)
        else:
            raise FileNotFoundError(f"No {kind} file matches: {item}")
    paths = list(dict.fromkeys(os.path.abspath(path) for path in paths))
    if not paths:
        raise ValueError(f"At least one {kind} file is required.")
    return paths


def infer_year(path, frame):
    if "time" in frame.columns:
        times = pd.to_datetime(frame["time"], errors="coerce")
        years = sorted({int(v) for v in times.dt.year.dropna().unique()})
        if len(years) == 1:
            return str(years[0])
        if years:
            return "_".join(map(str, years))
    match = re.search(r"(?:19|20)\d{2}", os.path.basename(path))
    return match.group(0) if match else Path(path).stem


def load_results(csv_paths):
    frames = []
    for path in csv_paths:
        frame = pd.read_csv(path)
        missing = sorted(REQUIRED_COLUMNS - set(frame.columns))
        if missing:
            raise ValueError(f"{path} lacks required columns: {missing}")
        frame = frame.copy()
        frame["_source_csv"] = os.path.basename(path)
        frame["_year"] = infer_year(path, frame)
        frames.append(frame)
    df = pd.concat(frames, ignore_index=True, sort=False)
    if df.empty:
        raise ValueError("Input deployment CSVs contain no rows.")
    if "time" in df.columns:
        dup = df.duplicated(["station", "time"], keep=False)
        if dup.any():
            examples = df.loc[dup, ["station", "time", "_source_csv"]].head(8)
            raise ValueError(
                "Duplicate (station, time) rows across inputs; refusing to double-count. "
                f"Examples: {examples.to_dict(orient='records')}"
            )
    return df


def read_test_stations(path):
    result = set()
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line and not line.startswith("#"):
                result.add(line.split()[0])
    if not result:
        raise ValueError(f"No station IDs found in {path}")
    return result


def metric_dict(pred, truth):
    pred = np.asarray(pred, dtype=float)
    truth = np.asarray(truth, dtype=float)
    if len(truth) == 0:
        raise ValueError("Cannot calculate metrics on zero samples.")
    rmse = float(np.sqrt(mean_squared_error(truth, pred)))
    abs_mean = float(np.mean(np.abs(truth)))
    return {
        "N": int(len(truth)),
        "RMSE": rmse,
        "MAE": float(mean_absolute_error(truth, pred)),
        "R2": float(r2_score(truth, pred)) if len(truth) >= 2 else np.nan,
        "Bias": float(np.mean(pred - truth)),
        "RelRMSE_pct": float(rmse / abs_mean * 100.0) if abs_mean > 0 else np.nan,
    }


def common_rows(sub):
    valid = sub["pwv_true"].notna() & sub[list(COLS.values())].notna().all(axis=1)
    return sub.loc[valid].copy()


def regime_metrics(sub):
    rows = []
    for regime in REGIMES:
        row = metric_dict(sub[COLS[regime]], sub["pwv_true"])
        row["model"] = regime
        rows.append(row)
    return pd.DataFrame(rows)[["model", "N", "RMSE", "MAE", "R2", "Bias", "RelRMSE_pct"]]


def per_station_metrics(sub):
    rows = []
    for station, group in sub.groupby("station", sort=True):
        for regime in REGIMES:
            row = metric_dict(group[COLS[regime]], group["pwv_true"])
            row.update({"station": station, "model": regime})
            rows.append(row)
    return pd.DataFrame(rows)[["station", "model", "N", "RMSE", "MAE", "R2", "Bias", "RelRMSE_pct"]]


def station_macro(per_station):
    if per_station.empty:
        return pd.DataFrame(columns=["model", "N_stations", "Mean_station_N", "RMSE_macro", "MAE_macro", "R2_macro", "Bias_macro"])
    return (
        per_station.groupby("model", sort=False)
        .agg(
            N_stations=("station", "nunique"),
            Mean_station_N=("N", "mean"),
            RMSE_macro=("RMSE", "mean"),
            MAE_macro=("MAE", "mean"),
            R2_macro=("R2", "mean"),
            Bias_macro=("Bias", "mean"),
        )
        .reset_index()
    )


def annual_metrics(sub):
    rows = []
    for year, group in sub.groupby("_year", sort=True):
        for row in regime_metrics(group).to_dict(orient="records"):
            row["year"] = year
            row["N_stations"] = int(group["station"].nunique())
            rows.append(row)
    return pd.DataFrame(rows)


def coverage_row(label, raw, common, requested_test_count):
    return {
        "population": label,
        "raw_samples": int(len(raw)),
        "common_samples": int(len(common)),
        "common_sample_rate_pct": float(100 * len(common) / len(raw)) if len(raw) else np.nan,
        "raw_stations": int(raw["station"].nunique()) if len(raw) else 0,
        "common_stations": int(common["station"].nunique()) if len(common) else 0,
        "requested_official36_stations": int(requested_test_count) if label == "official36" else np.nan,
        "official_station_coverage_pct": (
            float(100 * common["station"].nunique() / requested_test_count)
            if label == "official36" and requested_test_count else np.nan
        ),
    }


def summarize_audits(paths):
    """Flatten scalar audit fields, including one or more nested dictionaries."""
    def flatten(prefix, value, out):
        if isinstance(value, dict):
            for child_key, child_value in value.items():
                child_prefix = f"{prefix}_{child_key}" if prefix else str(child_key)
                flatten(child_prefix, child_value, out)
        elif isinstance(value, (str, int, float, bool)) or value is None:
            out[prefix] = value

    rows = []
    for path in paths:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, dict):
            raise ValueError(f"Audit JSON must contain an object: {path}")
        row = {"audit_file": os.path.basename(path)}
        for key, value in payload.items():
            flatten(str(key), value, row)
        rows.append(row)
    return pd.DataFrame(rows)


def make_plot(metrics_frame, label, out_path, title_label):
    names = ["real\n(oracle)", "real+P1 surf", "clim", "clim+P1 surf", "clim adj P1", "GPT3"]
    ordered = metrics_frame.set_index("model").loc[REGIMES]
    values = ordered["RMSE"].to_numpy()
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(names, values, color=["#2e8b57", "#66c2a5", "#8da0cb", "#fc8d62", "#e78ac3", "#a6d854"])
    bump = max(values) * 0.01 if max(values) else 0.01
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + bump, f"{value:.3f}", ha="center", fontsize=9)
    ax.set_ylabel("PWV RMSE (mm)")
    ax.set_title(f"{label} Phase1->Phase2 deployment comparison ({title_label}, common N={int(ordered['N'].iloc[0])})")
    plt.xticks(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    args = parse_args()
    csv_paths = expand_paths(args.csv, "CSV")
    audit_paths = expand_paths(args.audit, "audit JSON") if args.audit else []
    df = load_results(csv_paths)
    test_ids = read_test_stations(args.test_stations)
    df["is_test36"] = df["station"].isin(test_ids)
    os.makedirs(args.out, exist_ok=True)

    inferred_years = sorted(df["_year"].astype(str).unique())
    label_year = args.year if args.year != "auto" else "_".join(inferred_years)
    print(f"Input rows: {len(df)}, stations: {df['station'].nunique()}, input files: {len(csv_paths)}")
    print(f"Analysis label: {label_year}; inferred yearly groups: {inferred_years}")

    coverage = []
    for label, raw in (("all_stations", df), ("official36", df.loc[df["is_test36"]])):
        common = common_rows(raw)
        coverage.append(coverage_row(label, raw, common, len(test_ids)))
        if common.empty:
            raise ValueError(
                f"{label} has zero common samples across pwv_true and all six regimes. "
                "Inspect deployment matching audit rather than reporting metrics."
            )
        metrics_frame = regime_metrics(common)
        per_station = per_station_metrics(common)
        macro = station_macro(per_station)
        annual = annual_metrics(common)
        metrics_frame.to_csv(Path(args.out) / f"metrics_{label}_{label_year}.csv", index=False, float_format="%.6f")
        per_station.to_csv(Path(args.out) / f"per_station_{label}_{label_year}.csv", index=False, float_format="%.6f")
        macro.to_csv(Path(args.out) / f"station_macro_{label}_{label_year}.csv", index=False, float_format="%.6f")
        annual.to_csv(Path(args.out) / f"annual_metrics_{label}_{label_year}.csv", index=False, float_format="%.6f")
        print(f"\n===== {label}: common N={len(common)}, stations={common['station'].nunique()} =====")
        print(metrics_frame.to_string(index=False))
        if not args.no_plot:
            make_plot(metrics_frame, label, Path(args.out) / f"compare_{label}_{label_year}.png", label_year)

    coverage_frame = pd.DataFrame(coverage)
    coverage_frame.to_csv(Path(args.out) / f"coverage_summary_{label_year}.csv", index=False, float_format="%.6f")
    if audit_paths:
        audit_frame = summarize_audits(audit_paths)
        audit_frame.to_csv(Path(args.out) / f"p1_match_audit_summary_{label_year}.csv", index=False)
        print(f"Audit summaries written for {len(audit_frame)} file(s).")
    print(f"\nDone -> {args.out}")


if __name__ == "__main__":
    main()
