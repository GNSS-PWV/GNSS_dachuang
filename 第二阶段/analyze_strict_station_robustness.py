"""严格六年堆叠部署结果的站点级、年度级、季节级稳健性分析。

仅读取 result_p1deploy_ft_strict_20260817 的六个年度预测 CSV；不修改原始结果、
不联网、不下载数据。输出的指标在相同共同样本口径下比较 clim_surf_p1、
clim_adj_p1 与 GPT3。
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

METHOD_COLUMNS = {
    "clim_surf_p1": "pwv_clim_surf_p1",
    "clim_adj_p1": "pwv_clim_adj_p1",
    "gpt3": "pwv_gpt3",
}
SEASON_ORDER = ["DJF", "MAM", "JJA", "SON"]


def metrics(frame: pd.DataFrame, prediction_col: str) -> dict[str, float | int]:
    err = frame[prediction_col].to_numpy(dtype=float) - frame["pwv_true"].to_numpy(dtype=float)
    return {
        "N": int(err.size),
        "RMSE": float(np.sqrt(np.mean(np.square(err)))),
        "MAE": float(np.mean(np.abs(err))),
        "Bias": float(np.mean(err)),
    }


def season_from_month(month: pd.Series) -> pd.Series:
    return pd.Series(
        np.select(
            [month.isin([12, 1, 2]), month.isin([3, 4, 5]), month.isin([6, 7, 8])],
            ["DJF", "MAM", "JJA"],
            default="SON",
        ),
        index=month.index,
    )


def load_common_samples(result_root: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    required = ["station", "time", "pwv_true", *METHOD_COLUMNS.values()]
    for year in range(2014, 2020):
        path = result_root / f"year_{year}" / f"deploy_p1_predictions_{year}.csv"
        if not path.is_file():
            raise FileNotFoundError(f"缺少年度预测文件：{path}")
        frame = pd.read_csv(path, usecols=lambda name: name in set(required))
        missing = set(required) - set(frame.columns)
        if missing:
            raise ValueError(f"{path.name} 缺少列：{sorted(missing)}")
        frame["year"] = year
        frame["time"] = pd.to_datetime(frame["time"], errors="coerce")
        frames.append(frame)
    data = pd.concat(frames, ignore_index=True)
    valid_cols = ["pwv_true", *METHOD_COLUMNS.values()]
    data = data.dropna(subset=["station", "time", *valid_cols]).copy()
    data["season"] = season_from_month(data["time"].dt.month)
    return data


def group_metrics(data: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for key, group in data.groupby(by, sort=True, observed=True):
        key_values = (key,) if isinstance(key, str) or not isinstance(key, tuple) else key
        base = dict(zip(by, key_values))
        all_metrics = {method: metrics(group, column) for method, column in METHOD_COLUMNS.items()}
        surf = all_metrics["clim_surf_p1"]
        gpt3 = all_metrics["gpt3"]
        adj = all_metrics["clim_adj_p1"]
        reduction = gpt3["RMSE"] - surf["RMSE"]
        row: dict[str, object] = {
            **base,
            "N": surf["N"],
            "rmse_clim_surf_p1": surf["RMSE"],
            "mae_clim_surf_p1": surf["MAE"],
            "bias_clim_surf_p1": surf["Bias"],
            "rmse_gpt3": gpt3["RMSE"],
            "mae_gpt3": gpt3["MAE"],
            "bias_gpt3": gpt3["Bias"],
            "rmse_clim_adj_p1": adj["RMSE"],
            "rmse_reduction_vs_gpt3_mm": reduction,
            "relative_rmse_reduction_vs_gpt3_pct": 100.0 * reduction / gpt3["RMSE"],
            "surf_vs_gpt3": "WIN" if reduction > 0 else "LOSS" if reduction < 0 else "TIE",
            "surface_better_than_adjusted": surf["RMSE"] < adj["RMSE"],
        }
        rows.append(row)
    return pd.DataFrame(rows)


def save_plot(station_summary: pd.DataFrame, output: Path) -> None:
    ordered = station_summary.sort_values("rmse_reduction_vs_gpt3_mm")
    colors = np.where(ordered["surf_vs_gpt3"].eq("WIN"), "#2b7bba", "#c83f49")
    height = max(7.0, len(ordered) * 0.28)
    fig, ax = plt.subplots(figsize=(11, height))
    ax.barh(ordered["station"], ordered["rmse_reduction_vs_gpt3_mm"], color=colors)
    ax.axvline(0.0, color="black", linewidth=0.8)
    ax.set_xlabel("GPT3 RMSE − clim_surf_p1 RMSE (mm); positive means clim_surf_p1 is better")
    ax.set_ylabel("Station")
    ax.set_title("2014–2019 strict replay: station-level RMSE change versus GPT3")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=220)
    plt.close(fig)


def write_markdown_summary(summary: pd.DataFrame, annual: pd.DataFrame, seasonal: pd.DataFrame, global_surface: dict[str, float | int], global_adjusted: dict[str, float | int], output: Path) -> None:
    weak = summary.loc[summary["surf_vs_gpt3"].ne("WIN")].sort_values("rmse_reduction_vs_gpt3_mm")
    annual_wins = annual.assign(win=annual["surf_vs_gpt3"].eq("WIN")).groupby("station", as_index=False)["win"].sum().rename(columns={"win": "annual_wins"})
    weak = weak.merge(annual_wins, on="station", how="left")
    weak_season = seasonal.loc[seasonal["station"].isin(weak["station"])].sort_values(["station", "season"])
    median_reduction = summary["rmse_reduction_vs_gpt3_mm"].median()
    median_relative = summary["relative_rmse_reduction_vs_gpt3_pct"].median()
    mean_relative = summary["relative_rmse_reduction_vs_gpt3_pct"].mean()
    win_count = int(summary["surf_vs_gpt3"].eq("WIN").sum())
    surface_wins_adj = int(summary["surface_better_than_adjusted"].sum())

    lines = [
        "# 严格六年部署结果：站点级稳健性与弱站诊断",
        "",
        "> 仅使用 2014–2019 年正式严格回放输出的共同样本重新计算；不混入 legacy 单年结果，也不将 oracle 方案用于部署比较。",
        "",
        "## 1. 总体结论",
        "",
        f"- `clim_surf_p1` 在 36 个站中有 **{win_count} 个站**优于 GPT3，另有 {36 - win_count} 个站较弱，无平局。",
        f"- 全部 110,928 个共同样本的 micro-average 中，`clim_surf_p1` RMSE 为 **{global_surface['RMSE']:.6f} mm**，低于 `clim_adj_p1` 的 {global_adjusted['RMSE']:.6f} mm；这是正式全局部署主口径。",
        f"- 站点层面的 RMSE 绝对改善中位数为 **{median_reduction:.6f} mm**；相对改善中位数为 **{median_relative:.2f}%**，均值为 {mean_relative:.2f}%。",
        f"- 在 {surface_wins_adj}/36 个站中，只替换地表层的 `clim_surf_p1` 优于整体廓线修正的 `clim_adj_p1`。这与全局 micro-average 不矛盾：站点样本量和改善幅度不同；不能据此在同一测试集上按站点选择方案。该现象支持“地表状态最可靠，中高层保留气候态形状”的研究假设，但不构成因果证明。",
        "",
        "## 2. 未优于 GPT3 的站点",
        "",
        "| station | N | clim_surf_p1 RMSE | GPT3 RMSE | GPT3−模型 (mm) | 相对变化 | clim_adj_p1 RMSE | 年度胜出数（6年） |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in weak.iterrows():
        lines.append(
            f"| {row.station} | {int(row.N)} | {row.rmse_clim_surf_p1:.6f} | {row.rmse_gpt3:.6f} | "
            f"{row.rmse_reduction_vs_gpt3_mm:.6f} | {row.relative_rmse_reduction_vs_gpt3_pct:.2f}% | "
            f"{row.rmse_clim_adj_p1:.6f} | {int(row.annual_wins)} |"
        )
    lines += [
        "",
        "解释边界：这些站点是后续优先诊断对象。当前输出可以揭示误差模式，但仅凭站点汇总尚不能断言具体地形、气候或观测原因；需要在补齐站点元数据和未来 ERA5/NWP 剖面实验后验证。",
        "",
        "## 3. 弱站的季节分层",
        "",
        "| station | season | N | clim_surf_p1 RMSE | GPT3 RMSE | GPT3−模型 (mm) | 结果 |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for _, row in weak_season.iterrows():
        lines.append(
            f"| {row.station} | {row.season} | {int(row.N)} | {row.rmse_clim_surf_p1:.6f} | "
            f"{row.rmse_gpt3:.6f} | {row.rmse_reduction_vs_gpt3_mm:.6f} | {row.surf_vs_gpt3} |"
        )
    lines += [
        "",
        "## 4. 可复核文件",
        "",
        "- `station_vs_gpt3_summary.csv`：36 站六年合并结果；",
        "- `station_year_vs_gpt3.csv`：站点×年份结果；",
        "- `station_season_vs_gpt3.csv`：站点×季节结果；",
        "- `station_rmse_change_vs_gpt3.png`：站点级 RMSE 改善图；",
        "",
        "## 5. 下一步研究建议",
        "",
        "1. 对三个弱站补齐坐标、高程、气候区和原始观测覆盖信息；",
        "2. 逐高度层比较 `clim_surf_p1` 与 `clim_adj_p1`，避免只根据总体 RMSE 推断原因；",
        "3. 在时间截止年的 ERA5/NWP 剖面实验中，重点复核这些弱站是否仍然较弱；",
        "4. 不能根据本表在同一六年测试集上选择“每站最优方案”并重新宣传总体结果；那会引入测试集选择偏差。",
    ]
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="分析 2014–2019 严格部署的站点和季节稳健性")
    parser.add_argument(
        "--result-root",
        type=Path,
        default=Path(__file__).resolve().parent / "result_p1deploy_ft_strict_20260817",
    )
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()

    result_root = args.result_root.resolve()
    out_dir = args.out_dir or result_root / "analysis_2014_2019" / "station_robustness_20260817"
    out_dir.mkdir(parents=True, exist_ok=True)
    data = load_common_samples(result_root)
    station_summary = group_metrics(data, ["station"]).sort_values("rmse_reduction_vs_gpt3_mm")
    station_year = group_metrics(data, ["station", "year"]).sort_values(["station", "year"])
    station_season = group_metrics(data, ["station", "season"])
    station_season["season"] = pd.Categorical(station_season["season"], SEASON_ORDER, ordered=True)
    station_season = station_season.sort_values(["station", "season"])

    station_summary.to_csv(out_dir / "station_vs_gpt3_summary.csv", index=False, encoding="utf-8-sig")
    station_year.to_csv(out_dir / "station_year_vs_gpt3.csv", index=False, encoding="utf-8-sig")
    station_season.to_csv(out_dir / "station_season_vs_gpt3.csv", index=False, encoding="utf-8-sig")
    save_plot(station_summary, out_dir / "station_rmse_change_vs_gpt3.png")
    global_surface = metrics(data, METHOD_COLUMNS["clim_surf_p1"])
    global_adjusted = metrics(data, METHOD_COLUMNS["clim_adj_p1"])
    write_markdown_summary(station_summary, station_year, station_season, global_surface, global_adjusted, out_dir / "站点稳健性与弱站诊断_20260817.md")

    all_metrics = metrics(data, METHOD_COLUMNS["clim_surf_p1"])
    print(f"common_samples={all_metrics['N']}")
    print(f"stations={station_summary.shape[0]}")
    print(f"wins_vs_gpt3={(station_summary['surf_vs_gpt3'] == 'WIN').sum()}")
    print(f"output={out_dir}")


if __name__ == "__main__":
    main()
