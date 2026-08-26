"""Generate presentation-ready figures for the verified 2014-2019 strict replay result.

This utility is intentionally read-only with respect to model predictions and input CSVs.
It creates a separate figure bundle and explicitly distinguishes oracle analyses from
actual deployment candidates.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent
RESULT_DIR = ROOT / "result_p1deploy_ft_strict_20260817" / "analysis_2014_2019"
DEFAULT_OUTPUT_DIR = RESULT_DIR / "paper_ppt_figures_20260817"
METRICS_CSV = RESULT_DIR / "metrics_official36_2014_2019.csv"
ANNUAL_CSV = RESULT_DIR / "annual_metrics_official36_2014_2019.csv"
STATION_CSV = RESULT_DIR / "station_robustness_20260817" / "station_vs_gpt3_summary.csv"
COVERAGE_CSV = RESULT_DIR / "station_robustness_20260817" / "annual_station_coverage_strict.csv"
EXPECTED_YEARS = list(range(2014, 2020))
EXPECTED_MODELS = ["real", "real_surf_p1", "clim", "clim_surf_p1", "clim_adj_p1", "gpt3"]


class InputValidationError(ValueError):
    """Raised when a source table does not match the frozen strict-result scope."""


def require_columns(frame: pd.DataFrame, columns: Iterable[str], source: Path) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise InputValidationError(f"{source} is missing required columns: {missing}")


def require_exact_values(actual: Iterable, expected: Iterable, label: str) -> None:
    actual_values = list(actual)
    expected_values = list(expected)
    if actual_values != expected_values:
        raise InputValidationError(f"{label} expected {expected_values}, got {actual_values}")


def load_and_validate_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    for source in (METRICS_CSV, ANNUAL_CSV, STATION_CSV, COVERAGE_CSV):
        if not source.is_file():
            raise FileNotFoundError(f"Required source file is absent: {source}")

    metrics = pd.read_csv(METRICS_CSV)
    annual = pd.read_csv(ANNUAL_CSV)
    station = pd.read_csv(STATION_CSV)
    coverage = pd.read_csv(COVERAGE_CSV)

    require_columns(metrics, ["model", "N", "RMSE", "MAE", "Bias"], METRICS_CSV)
    require_columns(annual, ["model", "N", "RMSE", "year", "N_stations"], ANNUAL_CSV)
    require_columns(
        station,
        ["station", "N", "rmse_clim_surf_p1", "rmse_gpt3", "rmse_reduction_vs_gpt3_mm", "surf_vs_gpt3"],
        STATION_CSV,
    )
    require_columns(coverage, ["year", "stations_with_samples", "common_pwv_N"], COVERAGE_CSV)

    require_exact_values(metrics["model"].tolist(), EXPECTED_MODELS, "metric model order")
    require_exact_values(sorted(annual["year"].unique().tolist()), EXPECTED_YEARS, "annual years")
    require_exact_values(sorted(coverage["year"].unique().tolist()), EXPECTED_YEARS, "coverage years")
    if annual.groupby("year")["model"].nunique().ne(len(EXPECTED_MODELS)).any():
        raise InputValidationError("Every annual row must contain all six frozen model labels")
    if metrics["N"].nunique() != 1 or int(metrics["N"].iloc[0]) != 110_928:
        raise InputValidationError("Global metrics must describe 110,928 common samples")
    if len(station) != 36:
        raise InputValidationError(f"Station summary must contain 36 union stations, got {len(station)}")
    expected_station_outcomes = {"WIN": 33, "LOSS": 3}
    actual_station_outcomes = station["surf_vs_gpt3"].value_counts(dropna=False).to_dict()
    if actual_station_outcomes != expected_station_outcomes:
        raise InputValidationError(
            "Station outcome labels must be exactly WIN=33 and LOSS=3 (with no TIE or missing labels); "
            f"got {actual_station_outcomes}"
        )
    if coverage["stations_with_samples"].max() > 36:
        raise InputValidationError("Yearly stations cannot exceed the 36-station cross-year union")
    return metrics, annual, station, coverage


def save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> list[Path]:
    paths: list[Path] = []
    for suffix in ("png", "pdf", "svg"):
        target = output_dir / f"{stem}.{suffix}"
        kwargs = {"dpi": 240} if suffix == "png" else {}
        fig.savefig(target, bbox_inches="tight", **kwargs)
        paths.append(target)
    plt.close(fig)
    return paths


def global_rmse_figure(metrics: pd.DataFrame, output_dir: Path) -> list[Path]:
    labels = {
        "real": "Real profile\n(oracle)",
        "real_surf_p1": "Real profile + P1 surface\n(oracle)",
        "clim": "Climate profile",
        "clim_surf_p1": "Climate + P1 surface\n(deployment candidate)",
        "clim_adj_p1": "Climate + P1 full-profile\nadjustment",
        "gpt3": "GPT3 baseline",
    }
    colors = {
        "real": "#BDBDBD",
        "real_surf_p1": "#969696",
        "clim": "#56B4E9",
        "clim_surf_p1": "#0072B2",
        "clim_adj_p1": "#E69F00",
        "gpt3": "#D55E00",
    }
    hatches = {"real": "//", "real_surf_p1": "//", "clim": "", "clim_surf_p1": "", "clim_adj_p1": "", "gpt3": ""}
    ordered = metrics.set_index("model").loc[EXPECTED_MODELS].reset_index()
    fig, ax = plt.subplots(figsize=(11.0, 5.8), constrained_layout=True)
    bars = ax.bar(
        range(len(ordered)), ordered["RMSE"],
        color=[colors[m] for m in ordered["model"]],
        hatch=[hatches[m] for m in ordered["model"]],
        edgecolor="#333333", linewidth=0.6,
    )
    ax.set_title("2014–2019 historical replay evaluation: PWV RMSE comparison")
    ax.set_ylabel("RMSE (mm)")
    ax.set_xticks(range(len(ordered)), [labels[m] for m in ordered["model"]])
    ax.set_ylim(0, max(ordered["RMSE"]) * 1.23)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    for bar, value in zip(bars, ordered["RMSE"]):
        ax.annotate(f"{value:.3f}", (bar.get_x() + bar.get_width() / 2, value),
                    xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", fontsize=9)
    ax.text(
        0.01, 0.97,
        "Hatched bars are oracle analyses only; they are not deployment accuracies.",
        transform=ax.transAxes, ha="left", va="top", fontsize=9,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": "#777777", "alpha": 0.92},
    )
    return save_figure(fig, output_dir, "01_global_rmse_comparison_2014_2019")


def annual_rmse_figure(annual: pd.DataFrame, output_dir: Path) -> list[Path]:
    series = {
        "clim_surf_p1": ("Climate + P1 surface (deployment candidate)", "#0072B2"),
        "clim_adj_p1": ("Climate + P1 full-profile adjustment", "#E69F00"),
        "gpt3": ("GPT3 baseline", "#D55E00"),
    }
    fig, ax = plt.subplots(figsize=(9.5, 5.7), constrained_layout=True)
    for model, (label, color) in series.items():
        subset = annual.loc[annual["model"].eq(model)].sort_values("year")
        ax.plot(subset["year"], subset["RMSE"], marker="o", markersize=6, linewidth=2.2, color=color, label=label)
    ax.set_title("2014–2019 historical replay evaluation: annual PWV RMSE")
    ax.set_xlabel("Evaluation year")
    ax.set_ylabel("RMSE (mm)")
    ax.set_xticks(EXPECTED_YEARS)
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, loc="upper left")
    ax.text(0.99, 0.03, "All shown series are non-oracle comparisons.", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=8.8)
    return save_figure(fig, output_dir, "02_annual_rmse_2014_2019")


def station_reduction_figure(station: pd.DataFrame, output_dir: Path) -> list[Path]:
    ordered = station.sort_values("rmse_reduction_vs_gpt3_mm", ascending=True).reset_index(drop=True)
    values = ordered["rmse_reduction_vs_gpt3_mm"]
    colors = ["#D55E00" if status == "LOSS" else "#0072B2" for status in ordered["surf_vs_gpt3"]]
    fig_height = max(7.0, 0.35 * len(ordered) + 1.6)
    fig, ax = plt.subplots(figsize=(10.5, fig_height), constrained_layout=True)
    bars = ax.barh(ordered["station"], values, color=colors, edgecolor="#333333", linewidth=0.35)
    ax.axvline(0, color="#333333", linewidth=0.9)
    ax.set_title("Station-level RMSE difference: GPT3 − Climate + P1 surface")
    ax.set_xlabel("RMSE reduction versus GPT3 (mm); positive means Climate + P1 surface is better")
    ax.set_ylabel("Station")
    ax.grid(axis="x", alpha=0.25)
    ax.set_axisbelow(True)
    for bar, value, status in zip(bars, values, ordered["surf_vs_gpt3"]):
        x = value + (0.003 if value >= 0 else -0.003)
        ha = "left" if value >= 0 else "right"
        ax.text(x, bar.get_y() + bar.get_height() / 2, f"{value:+.3f}", va="center", ha=ha, fontsize=7.2)
        if status == "LOSS":
            ax.text(ax.get_xlim()[0] if ax.get_xlim()[0] < value else value, bar.get_y() + bar.get_height() / 2, "", fontsize=1)
    ax.text(0.01, 0.985, "Blue: WIN (33 stations); red: LOSS (3 exploratory fixed stations).",
            transform=ax.transAxes, ha="left", va="top", fontsize=9,
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": "#777777", "alpha": 0.92})
    return save_figure(fig, output_dir, "03_station_rmse_reduction_vs_gpt3_2014_2019")


def coverage_figure(coverage: pd.DataFrame, output_dir: Path) -> list[Path]:
    ordered = coverage.sort_values("year")
    fig, ax_left = plt.subplots(figsize=(9.5, 5.6), constrained_layout=True)
    bars = ax_left.bar(ordered["year"].astype(str), ordered["common_pwv_N"], color="#56B4E9", label="Common samples")
    ax_left.set_title("Actual yearly comparable coverage (36 stations is the cross-year union)")
    ax_left.set_xlabel("Evaluation year")
    ax_left.set_ylabel("Common PWV samples")
    ax_left.grid(axis="y", alpha=0.25)
    ax_left.set_axisbelow(True)
    for bar, value in zip(bars, ordered["common_pwv_N"]):
        ax_left.annotate(f"{int(value):,}", (bar.get_x() + bar.get_width() / 2, value), xytext=(0, 4),
                         textcoords="offset points", ha="center", va="bottom", fontsize=8)
    ax_right = ax_left.twinx()
    ax_right.plot(ordered["year"].astype(str), ordered["stations_with_samples"], color="#009E73", marker="o",
                  markersize=6, linewidth=2.2, label="Stations with samples")
    ax_right.set_ylabel("Stations with samples")
    ax_right.set_ylim(0, 36)
    ax_right.set_yticks(range(0, 37, 6))
    handles_a, labels_a = ax_left.get_legend_handles_labels()
    handles_b, labels_b = ax_right.get_legend_handles_labels()
    ax_left.legend(handles_a + handles_b, labels_a + labels_b, frameon=False, loc="upper left")
    return save_figure(fig, output_dir, "04_yearly_coverage_2014_2019")


def write_captions(output_dir: Path) -> Path:
    caption = """# 论文 / PPT 图表与图注（2014–2019 严格历史回放式评估）

> 本目录由 `第二阶段/generate_paper_ppt_figures.py` 自动生成，只读取已冻结的严格结果 CSV；未下载、未写入或重新训练任何模型数据。

## 共同口径

- 评估范围：2014–2019，110,928 个共同样本；36 个站仅表示跨年度联合覆盖，单年度实际有样本站点为 27–30。
- 当前最佳部署候选是 `clim_surf_p1`：以第一阶段预测的 PS/TS/WPS 替换气候态剖面的地表状态。
- `real` 与 `real_surf_p1` 均为 oracle 分析，不能表述为实际部署精度。
- 本图包是历史回放式堆叠部署评估，不是完全自治实时系统，也不是训练截止年后的未来独立验证。
- 弱站/输入关联只能作探索性描述，不作因果归因或未来泛化承诺。

## 图 1：全局 RMSE 对比

**建议图注：** 2014–2019 年严格历史回放式堆叠部署评估的 PWV RMSE 对比（110,928 个共同样本）。`clim_surf_p1` 的 RMSE 为 0.242729 mm，GPT3 为 0.346177 mm；前者较 GPT3 低 0.103448 mm（29.88%）。带斜线的 `real` 和 `real_surf_p1` 为 oracle 分析，仅用于解释上界与地表状态误差影响，不代表可部署精度。

## 图 2：年度 RMSE

**建议图注：** 2014–2019 年逐年 PWV RMSE。图中只展示可比较的非-oracle 方案：`clim_surf_p1`、`clim_adj_p1` 和 GPT3。`clim_surf_p1` 在当前六个年度均优于 GPT3；该结论限于本历史回放评估范围。

## 图 3：站点级 GPT3 差值

**建议图注：** 官方 36 站跨年度联合覆盖上的站点级 RMSE 差值，定义为 GPT3 RMSE − `clim_surf_p1` RMSE。正值表示 `clim_surf_p1` 优于 GPT3。当前评估期内 33/36 站为正值；COM00080222、SFM00068842 和 SAM00040417 为固定的探索性复核弱站。该图不用于归因，也不用于按测试集为单站选择方案。

## 图 4：年度可比覆盖

**建议图注：** 严格共同样本筛选后每个年度的实际可比样本数及有样本站点数。36 个站是六年联合覆盖；年度实际覆盖为 27–30 站，因此不能将 36 个站描述为每年稳定覆盖。

## 文件格式

每张图同时输出 PNG（适合 PPT）、PDF 和 SVG（适合论文排版）。文件名中的 `2014_2019` 仅表示本冻结历史评估范围，不表示未来验证。
"""
    target = output_dir / "论文PPT图表与图注_20260817.md"
    target.write_text(caption, encoding="utf-8")
    return target


def verify_outputs(paths: Iterable[Path]) -> None:
    for path in paths:
        if not path.is_file() or path.stat().st_size < 1024:
            raise RuntimeError(f"Figure output is missing or unexpectedly small: {path}")
    png_paths = [path for path in paths if path.suffix.lower() == ".png"]
    for path in png_paths:
        image = plt.imread(path)
        if image.ndim not in (2, 3) or image.shape[0] < 300 or image.shape[1] < 300:
            raise RuntimeError(f"PNG output has invalid dimensions: {path} -> {image.shape}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for the new figure bundle")
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics, annual, station, coverage = load_and_validate_inputs()
    outputs: list[Path] = []
    outputs.extend(global_rmse_figure(metrics, output_dir))
    outputs.extend(annual_rmse_figure(annual, output_dir))
    outputs.extend(station_reduction_figure(station, output_dir))
    outputs.extend(coverage_figure(coverage, output_dir))
    captions = write_captions(output_dir)
    outputs.append(captions)
    verify_outputs([path for path in outputs if path.suffix.lower() != ".md"])

    report = {
        "status": "PASS",
        "scope": "2014-2019 strict historical replay evaluation",
        "global_common_samples": int(metrics["N"].iloc[0]),
        "cross_year_union_stations": int(len(station)),
        "actual_yearly_station_range": [int(coverage["stations_with_samples"].min()), int(coverage["stations_with_samples"].max())],
        "loss_stations": station.loc[station["surf_vs_gpt3"].eq("LOSS"), "station"].tolist(),
        "output_files": [path.name for path in outputs],
    }
    report_path = output_dir / "figure_generation_report_20260817.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()

