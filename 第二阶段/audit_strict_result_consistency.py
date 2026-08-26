"""Reproducible consistency audit for the strict 2014-2019 P1 deployment results.

Reads only existing analysis CSV files. It does not download data, train models, or
modify prediction files. The audit avoids two known pitfalls: the RMSE reduction
is defined as GPT3 minus clim_surf_p1, and floating-point comparisons use a
tolerance instead of direct equality across independently written CSV files.
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

MODELS = {"real", "real_surf_p1", "clim", "clim_surf_p1", "clim_adj_p1", "gpt3"}
YEARS = set(range(2014, 2020))
INPUT_KEYS = ("ps", "ts", "wps", "tm")
ABS_TOL = 1e-12
REL_TOL = 1e-10


@dataclass(frozen=True)
class AuditPaths:
    analysis_dir: Path
    robustness_dir: Path

    @property
    def metrics(self) -> Path:
        return self.analysis_dir / "metrics_official36_2014_2019.csv"

    @property
    def annual_metrics(self) -> Path:
        return self.analysis_dir / "annual_metrics_official36_2014_2019.csv"

    @property
    def coverage(self) -> Path:
        return self.analysis_dir / "coverage_summary_2014_2019.csv"

    @property
    def annual_coverage(self) -> Path:
        return self.robustness_dir / "annual_station_coverage_strict.csv"

    @property
    def station_summary(self) -> Path:
        return self.robustness_dir / "station_vs_gpt3_summary.csv"

    @property
    def diagnostics(self) -> Path:
        return self.robustness_dir / "station_surface_input_diagnostics.csv"

    @property
    def rank_context(self) -> Path:
        return self.robustness_dir / "station_surface_input_error_rank_context.csv"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def close(a: float, b: float, *, abs_tol: float = ABS_TOL, rel_tol: float = REL_TOL) -> bool:
    return math.isclose(float(a), float(b), abs_tol=abs_tol, rel_tol=rel_tol)


def read_csv(path: Path, required: Iterable[str]) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Missing audit input: {path}")
    frame = pd.read_csv(path)
    missing = set(required) - set(frame.columns)
    require(not missing, f"{path.name} missing columns: {sorted(missing)}")
    require(not frame.empty, f"{path.name} is empty")
    return frame


def finite(frame: pd.DataFrame, columns: Iterable[str], source: str) -> None:
    for column in columns:
        values = pd.to_numeric(frame[column], errors="coerce")
        require(values.notna().all(), f"{source}.{column} has NaN/non-numeric values")
        require(values.map(math.isfinite).all(), f"{source}.{column} has non-finite values")


def as_bool(value: object) -> bool:
    text = str(value).strip().lower()
    if text == "true":
        return True
    if text == "false":
        return False
    raise AssertionError(f"Invalid boolean encoding: {value!r}")


def verify_metrics(paths: AuditPaths) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metrics = read_csv(paths.metrics, {"model", "N", "RMSE", "MAE", "Bias"})
    annual = read_csv(paths.annual_metrics, {"model", "N", "RMSE", "MAE", "Bias", "year", "N_stations"})
    coverage = read_csv(paths.coverage, {"population", "raw_samples", "common_samples", "raw_stations", "common_stations"})
    require(len(metrics) == len(MODELS), f"Expected {len(MODELS)} global metric rows, got {len(metrics)}")
    require(set(metrics["model"]) == MODELS, "Global metrics model set differs from expected set")
    require(metrics["model"].is_unique, "Global metrics model names are duplicated")
    finite(metrics, ("N", "RMSE", "MAE", "Bias"), paths.metrics.name)
    common_n = int(metrics["N"].iloc[0])
    require((metrics["N"].astype(int) == common_n).all(), "Global metric models do not use one common N")

    require(set(annual["model"]) == MODELS, "Annual metric model set differs from expected set")
    require(set(annual["year"].astype(int)) == YEARS, "Annual metrics do not cover exactly 2014-2019")
    require(len(annual) == len(MODELS) * len(YEARS), "Annual metric row count is not 6 models x 6 years")
    require(not annual.duplicated(["model", "year"]).any(), "Annual metrics have duplicate model/year rows")
    finite(annual, ("N", "RMSE", "MAE", "Bias", "N_stations"), paths.annual_metrics.name)
    for model, group in annual.groupby("model", observed=True):
        require(int(group["N"].sum()) == common_n, f"{model}: annual N sum differs from global N")

    official = coverage.loc[coverage["population"].astype(str).eq("official36")]
    require(len(official) == 1, "Expected exactly one official36 coverage row")
    official_row = official.iloc[0]
    require(int(official_row["common_samples"]) == common_n, "Coverage common_samples differs from global N")
    require(int(official_row["common_stations"]) == 36, "Coverage common_stations is not 36")
    return metrics, annual, coverage


def verify_station_files(paths: AuditPaths, annual: pd.DataFrame) -> dict[str, int]:
    annual_coverage = read_csv(paths.annual_coverage, {"year", "stations_with_samples", "common_pwv_N"})
    summary = read_csv(
        paths.station_summary,
        {
            "station", "N", "rmse_clim_surf_p1", "rmse_gpt3", "rmse_clim_adj_p1",
            "rmse_reduction_vs_gpt3_mm", "relative_rmse_reduction_vs_gpt3_pct",
            "surf_vs_gpt3", "surface_better_than_adjusted",
        },
    )
    diagnostics = read_csv(
        paths.diagnostics,
        {
            "station", "pwv_common_N", "N_clim_surf_p1", "N_gpt3",
            "rmse_clim_surf_p1_mm", "rmse_gpt3_mm", "gpt3_minus_clim_surf_p1_rmse_mm",
            "surf_vs_gpt3", *(field for key in INPUT_KEYS for field in (f"N_{key}", f"rmse_{key}_p1")),
        },
    )
    ranks = read_csv(
        paths.rank_context,
        {
            "station", "pwv_common_N", "rmse_clim_surf_p1_mm", "rmse_gpt3_mm",
            "gpt3_minus_clim_surf_p1_rmse_mm", "surf_vs_gpt3",
            *(field for key in INPUT_KEYS for field in (f"rmse_{key}_p1", f"rmse_{key}_p1_rank_desc")),
        },
    )

    require(len(annual_coverage) == 6, "Annual coverage must have six rows")
    require(set(annual_coverage["year"].astype(int)) == YEARS, "Annual coverage years differ from 2014-2019")
    require(annual_coverage["year"].is_unique, "Annual coverage contains duplicate years")
    finite(annual_coverage, ("stations_with_samples", "common_pwv_N"), paths.annual_coverage.name)
    clim_annual = annual.loc[annual["model"].eq("clim_surf_p1")].set_index("year")
    for row in annual_coverage.itertuples(index=False):
        expected = clim_annual.loc[int(row.year)]
        require(int(row.common_pwv_N) == int(expected.N), f"{row.year}: annual coverage N differs from clim_surf_p1 N")
        require(int(row.stations_with_samples) == int(expected.N_stations), f"{row.year}: station count differs from annual metric")

    for name, frame in (("summary", summary), ("diagnostics", diagnostics), ("rank_context", ranks)):
        require(len(frame) == 36, f"{name} row count is not 36")
        require(frame["station"].is_unique, f"{name} has duplicate stations")
        require(frame["station"].notna().all(), f"{name} has empty station keys")
    station_set = set(summary["station"])
    require(station_set == set(diagnostics["station"]) == set(ranks["station"]), "Station sets differ across station CSVs")

    finite(summary, ("N", "rmse_clim_surf_p1", "rmse_gpt3", "rmse_clim_adj_p1", "rmse_reduction_vs_gpt3_mm", "relative_rmse_reduction_vs_gpt3_pct"), paths.station_summary.name)
    finite(diagnostics, ("pwv_common_N", "N_clim_surf_p1", "N_gpt3", "rmse_clim_surf_p1_mm", "rmse_gpt3_mm", "gpt3_minus_clim_surf_p1_rmse_mm", *(f"N_{key}" for key in INPUT_KEYS), *(f"rmse_{key}_p1" for key in INPUT_KEYS)), paths.diagnostics.name)
    finite(ranks, ("pwv_common_N", "rmse_clim_surf_p1_mm", "rmse_gpt3_mm", "gpt3_minus_clim_surf_p1_rmse_mm", *(f"rmse_{key}_p1" for key in INPUT_KEYS), *(f"rmse_{key}_p1_rank_desc" for key in INPUT_KEYS)), paths.rank_context.name)

    diag_by_station = diagnostics.set_index("station")
    rank_by_station = ranks.set_index("station")
    for row in summary.itertuples(index=False):
        diag = diag_by_station.loc[row.station]
        require(int(row.N) == int(diag.pwv_common_N), f"{row.station}: summary N differs from diagnostics N")
        require(close(row.rmse_clim_surf_p1, diag.rmse_clim_surf_p1_mm), f"{row.station}: surface RMSE differs across files")
        require(close(row.rmse_gpt3, diag.rmse_gpt3_mm), f"{row.station}: GPT3 RMSE differs across files")
        expected_reduction = float(row.rmse_gpt3) - float(row.rmse_clim_surf_p1)
        require(close(row.rmse_reduction_vs_gpt3_mm, expected_reduction), f"{row.station}: incorrect summary RMSE reduction sign/value")
        expected_relative = 100.0 * expected_reduction / float(row.rmse_gpt3)
        require(close(row.relative_rmse_reduction_vs_gpt3_pct, expected_relative, abs_tol=1e-10), f"{row.station}: incorrect relative RMSE reduction")
        expected_label = "WIN" if float(row.rmse_clim_surf_p1) < float(row.rmse_gpt3) else "LOSS"
        require(row.surf_vs_gpt3 == expected_label, f"{row.station}: invalid WIN/LOSS label")
        expected_adjusted = float(row.rmse_clim_surf_p1) < float(row.rmse_clim_adj_p1)
        require(as_bool(row.surface_better_than_adjusted) == expected_adjusted, f"{row.station}: invalid adjusted comparison flag")
        for key in INPUT_KEYS:
            require(int(diag.pwv_common_N) == int(diag[f"N_{key}"]), f"{row.station}: N_{key} differs from common N")
        rank = rank_by_station.loc[row.station]
        require(int(diag.pwv_common_N) == int(rank.pwv_common_N), f"{row.station}: rank-context N differs")
        require(close(diag.rmse_clim_surf_p1_mm, rank.rmse_clim_surf_p1_mm), f"{row.station}: surface RMSE differs from rank context")
        require(close(diag.rmse_gpt3_mm, rank.rmse_gpt3_mm), f"{row.station}: GPT3 RMSE differs from rank context")

    for key in INPUT_KEYS:
        values = ranks[f"rmse_{key}_p1"].astype(float)
        expected_ranks = values.rank(method="min", ascending=False).astype(int)
        observed_ranks = ranks[f"rmse_{key}_p1_rank_desc"].astype(int)
        require(expected_ranks.equals(observed_ranks), f"{key}: ranks do not match rank_context values")

    weak = int(summary["surf_vs_gpt3"].eq("LOSS").sum())
    return {
        "annual_rows": int(len(annual_coverage)),
        "max_annual_station_count": int(annual_coverage["stations_with_samples"].max()),
        "stations": int(len(summary)),
        "weak_stations": weak,
        "diagnostic_rows": int(len(diagnostics)),
        "rank_rows": int(len(ranks)),
    }


def main() -> None:
    default_dir = Path(__file__).resolve().parent / "result_p1deploy_ft_strict_20260817" / "analysis_2014_2019"
    parser = argparse.ArgumentParser(description="Audit strict 2014-2019 P1 deployment result consistency.")
    parser.add_argument("--analysis-dir", type=Path, default=default_dir)
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()

    analysis_dir = args.analysis_dir.resolve()
    paths = AuditPaths(analysis_dir=analysis_dir, robustness_dir=analysis_dir / "station_robustness_20260817")
    metrics, annual, _ = verify_metrics(paths)
    station_summary = verify_station_files(paths, annual)
    global_metrics = metrics.set_index("model")
    surface_rmse = float(global_metrics.loc["clim_surf_p1", "RMSE"])
    gpt3_rmse = float(global_metrics.loc["gpt3", "RMSE"])
    result = {
        "status": "PASS",
        "metrics_rows": int(len(metrics)),
        "common_samples": int(global_metrics.loc["clim_surf_p1", "N"]),
        "union_stations": station_summary["stations"],
        "annual_years": sorted(YEARS),
        "max_annual_station_count": station_summary["max_annual_station_count"],
        "weak_stations": station_summary["weak_stations"],
        "clim_surf_p1_rmse": surface_rmse,
        "gpt3_rmse": gpt3_rmse,
        "improvement_pct": 100.0 * (gpt3_rmse - surface_rmse) / gpt3_rmse,
        "diagnostic_rows": station_summary["diagnostic_rows"],
        "rank_rows": station_summary["rank_rows"],
        "assertion_notes": [
            "RMSE reduction is defined as GPT3 minus clim_surf_p1.",
            "Cross-file floating-point checks use explicit tolerance.",
            "Input-error ranks are recomputed from rank_context.csv itself.",
        ],
    }
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
