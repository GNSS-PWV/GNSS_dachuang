"""为严格六年弱站输入误差诊断补充全体站点相对排名上下文。

只读取 station_surface_input_diagnostics.csv；不联网、不下载、不改写原始预测。
输出是探索性描述，不能用于因果归因或测试集上的逐站方案选择。
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

METRICS = {
    "rmse_ps_p1": "PS P1 RMSE",
    "rmse_ts_p1": "TS P1 RMSE",
    "rmse_wps_p1": "WPS P1 RMSE",
    "rmse_tm_p1": "Tm P1 RMSE（仅诊断）",
}


def enrich_rank_context(data: pd.DataFrame) -> pd.DataFrame:
    result = data.copy()
    for metric in METRICS:
        rank_col = f"{metric}_rank_desc"
        result[rank_col] = result[metric].rank(ascending=False, method="min").astype("Int64")
    return result


def write_report(data: pd.DataFrame, output: Path) -> None:
    weak = data.loc[data["surf_vs_gpt3"].ne("WIN")].copy()
    total = len(data)
    lines = [
        "# 弱站输入误差的全体站点相对位置（探索性描述）",
        "",
        "## 1. 解释边界",
        "",
        "- 基于 `station_surface_input_diagnostics.csv` 的 36 个站点结果；只提供相对排名上下文。",
        "- 每项 `rank_desc` 按该输入 RMSE 降序排列，`1` 表示 36 站中该输入 RMSE 最大；相同数值采用并列最小名次。",
        "- 弱站定义仍仅来自 `clim_surf_p1` 与 GPT3 的 PWV RMSE 比较，输入误差排名没有参与弱站筛选。",
        "- 这是当前评估期的事后探索性分析；排名较高只能提示后续需要核验的误差模式，不能证明输入误差导致 PWV 弱站表现。Tm 仅作诊断，不是二阶段模型输入。",
        "",
        "## 2. 当前弱站的输入 RMSE 与排名",
        "",
        "| station | GPT3−clim_surf_p1 PWV RMSE (mm) | PS RMSE / rank | TS RMSE / rank | WPS RMSE / rank | Tm RMSE / rank（仅诊断） |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for _, row in weak.iterrows():
        lines.append(
            f"| {row.station} | {row.gpt3_minus_clim_surf_p1_rmse_mm:.6f} | "
            f"{row.rmse_ps_p1:.6f} / {int(row.rmse_ps_p1_rank_desc)}/{total} | "
            f"{row.rmse_ts_p1:.6f} / {int(row.rmse_ts_p1_rank_desc)}/{total} | "
            f"{row.rmse_wps_p1:.6f} / {int(row.rmse_wps_p1_rank_desc)}/{total} | "
            f"{row.rmse_tm_p1:.6f} / {int(row.rmse_tm_p1_rank_desc)}/{total} |"
        )
    lines += [
        "",
        "## 3. 使用限制",
        "",
        "- 后续训练截止年之后的独立实验应固定分析名单、指标和判定规则，同时报告全体预先定义站点与总体结果；",
        "- 不得据此在同一测试集逐站挑选二阶段方案，或将排名当作因果结论。",
    ]
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="输出弱站输入误差相对全体站点的探索性排名上下文")
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=Path(__file__).resolve().parent
        / "result_p1deploy_ft_strict_20260817"
        / "analysis_2014_2019"
        / "station_robustness_20260817"
        / "station_surface_input_diagnostics.csv",
    )
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()

    input_csv = args.input_csv.resolve()
    out_dir = args.out_dir or input_csv.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(input_csv, encoding="utf-8-sig")
    required = {"station", "surf_vs_gpt3", "gpt3_minus_clim_surf_p1_rmse_mm", *METRICS}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"输入 CSV 缺少列：{sorted(missing)}")
    context = enrich_rank_context(data)
    context.to_csv(out_dir / "station_surface_input_error_rank_context.csv", index=False, encoding="utf-8-sig")
    write_report(context, out_dir / "弱站输入误差全体站点相对位置_20260817.md")
    print(f"stations={len(context)}")
    print(f"weak_stations={(context['surf_vs_gpt3'] != 'WIN').sum()}")
    print(f"output={out_dir}")


if __name__ == "__main__":
    main()
