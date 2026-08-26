"""Leave-one-year-out robustness check for the fixed exploratory LOSS stations.

The script only reads existing station-level strict deployment summaries. It does
not download data, train models, or change any prediction/model file. Pooled
RMSE is recomputed from N * RMSE^2 for each retained year.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def pooled_rmse(frame: pd.DataFrame, column: str) -> float:
    weights = frame["N"].to_numpy(dtype=float)
    rmses = frame[column].to_numpy(dtype=float)
    return float(np.sqrt(np.sum(weights * np.square(rmses)) / np.sum(weights)))


def build_leave_one_year_out(yearly: pd.DataFrame, weak_stations: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []
    for station in weak_stations:
        station_years = yearly.loc[yearly["station"].eq(station)].sort_values("year").copy()
        if len(station_years) < 2:
            raise ValueError(f"{station}: fewer than two covered years; leave-one-year-out is undefined")
        station_rows: list[dict[str, object]] = []
        for dropped_year in station_years["year"].astype(int):
            kept = station_years.loc[station_years["year"].astype(int).ne(dropped_year)]
            surface = pooled_rmse(kept, "rmse_clim_surf_p1")
            gpt3 = pooled_rmse(kept, "rmse_gpt3")
            reduction = gpt3 - surface
            row = {
                "station": station,
                "dropped_year": dropped_year,
                "retained_years": ";".join(str(year) for year in kept["year"].astype(int)),
                "retained_N": int(kept["N"].sum()),
                "clim_surf_p1_pooled_rmse_mm": surface,
                "gpt3_pooled_rmse_mm": gpt3,
                "gpt3_minus_clim_surf_p1_rmse_mm": reduction,
                "surf_vs_gpt3": "WIN" if reduction > 0 else "LOSS" if reduction < 0 else "TIE",
            }
            rows.append(row)
            station_rows.append(row)
        changes = [float(row["gpt3_minus_clim_surf_p1_rmse_mm"]) for row in station_rows]
        summaries.append(
            {
                "station": station,
                "covered_years": ";".join(str(year) for year in station_years["year"].astype(int)),
                "leave_one_year_out_runs": len(station_rows),
                "all_runs_remain_loss": all(row["surf_vs_gpt3"] == "LOSS" for row in station_rows),
                "reduction_min_mm": min(changes),
                "reduction_max_mm": max(changes),
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(summaries)


def write_report(details: pd.DataFrame, summary: pd.DataFrame, output: Path) -> None:
    lines = [
        "# 探索性弱站：逐年留一稳健性复核（2026-08-17）",
        "",
        "> 本复核只使用既有 `station_year_vs_gpt3.csv`。每次去掉一个已有可比年份，再用剩余年份的 `N × RMSE²` 合并计算 pooled RMSE。它检验弱站的汇总结论是否可能由单一年份驱动；不用于因果归因或未来泛化声明。",
        "",
        "## 1. 汇总",
        "",
        "| station | 原有覆盖年份 | 留一次数 | 每次留一后均为 LOSS | GPT3−clim_surf_p1 差值范围（mm） |",
        "|---|---|---:|---|---:|",
    ]
    for row in summary.itertuples(index=False):
        lines.append(
            f"| {row.station} | {row.covered_years} | {int(row.leave_one_year_out_runs)} | "
            f"{bool(row.all_runs_remain_loss)} | {row.reduction_min_mm:.6f} 至 {row.reduction_max_mm:.6f} |"
        )
    lines += [
        "",
        "## 2. 逐次结果",
        "",
        "| station | 剔除年份 | 保留样本数 | clim_surf_p1 pooled RMSE | GPT3 pooled RMSE | GPT3−clim_surf_p1 | 标签 |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in details.itertuples(index=False):
        lines.append(
            f"| {row.station} | {int(row.dropped_year)} | {int(row.retained_N)} | "
            f"{row.clim_surf_p1_pooled_rmse_mm:.6f} | {row.gpt3_pooled_rmse_mm:.6f} | "
            f"{row.gpt3_minus_clim_surf_p1_rmse_mm:.6f} | {row.surf_vs_gpt3} |"
        )
    lines += [
        "",
        "## 3. 解释边界",
        "",
        "- `LOSS` 仅表示在该站、该留一年份后的共同可比样本中，`clim_surf_p1` pooled RMSE 未低于 GPT3。",
        "- 该检验不能识别地表输入误差、站点条件、气候背景或观测质量的因果作用。",
        "- 该检验仍在 2014–2019 既有评估期内进行，不能替代训练截止年之后的独立验证。",
    ]
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    default_dir = Path(__file__).resolve().parent / "result_p1deploy_ft_strict_20260817" / "analysis_2014_2019" / "station_robustness_20260817"
    parser = argparse.ArgumentParser(description="Leave-one-year-out review for fixed strict-deployment LOSS stations.")
    parser.add_argument("--input-dir", type=Path, default=default_dir)
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()

    input_dir = args.input_dir.resolve()
    out_dir = (args.out_dir or input_dir).resolve()
    yearly = pd.read_csv(input_dir / "station_year_vs_gpt3.csv")
    total = pd.read_csv(input_dir / "station_vs_gpt3_summary.csv")
    required_yearly = {"station", "year", "N", "rmse_clim_surf_p1", "rmse_gpt3"}
    required_total = {"station", "surf_vs_gpt3"}
    missing = required_yearly - set(yearly.columns)
    if missing:
        raise ValueError(f"station_year_vs_gpt3.csv missing columns: {sorted(missing)}")
    missing = required_total - set(total.columns)
    if missing:
        raise ValueError(f"station_vs_gpt3_summary.csv missing columns: {sorted(missing)}")
    weak = total.loc[total["surf_vs_gpt3"].eq("LOSS"), "station"].sort_values().tolist()
    if not weak:
        raise ValueError("No fixed LOSS station found in total station summary")
    details, summary = build_leave_one_year_out(yearly, weak)
    out_dir.mkdir(parents=True, exist_ok=True)
    details.to_csv(out_dir / "weak_station_leave_one_year_out_20260817.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(out_dir / "weak_station_leave_one_year_out_summary_20260817.csv", index=False, encoding="utf-8-sig")
    write_report(details, summary, out_dir / "弱站逐年留一稳健性复核_20260817.md")
    print(f"weak_stations={len(weak)}")
    print(f"runs={len(details)}")
    print(f"all_loss_after_each_leave_out={int(summary['all_runs_remain_loss'].sum())}/{len(summary)}")
    print(f"output={out_dir}")


if __name__ == "__main__":
    main()
