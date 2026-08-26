"""输出严格六年部署共同样本的年度站点覆盖口径。

只读取既有严格预测结果；不联网、不下载、不修改原始预测。
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from diagnose_strict_station_surface_inputs import load_common_samples

YEARS = list(range(2014, 2020))


def build_coverage(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    station_year = (
        data.groupby(["station", "year"], observed=True)
        .size()
        .rename("common_pwv_N")
        .reset_index()
    )
    stations = sorted(data["station"].unique())
    full_index = pd.MultiIndex.from_product([stations, YEARS], names=["station", "year"])
    matrix = (
        station_year.set_index(["station", "year"])["common_pwv_N"]
        .reindex(full_index, fill_value=0)
        .rename("common_pwv_N")
        .reset_index()
    )
    matrix["available"] = matrix["common_pwv_N"].gt(0)
    annual = (
        matrix.groupby("year", observed=True)
        .agg(
            stations_with_samples=("available", "sum"),
            common_pwv_N=("common_pwv_N", "sum"),
        )
        .reset_index()
    )
    station = (
        matrix.groupby("station", observed=True)
        .agg(years_with_samples=("available", "sum"), total_common_pwv_N=("common_pwv_N", "sum"))
        .reset_index()
        .sort_values(["years_with_samples", "station"], ascending=[True, True])
    )
    return annual, station, matrix


def write_report(annual: pd.DataFrame, station: pd.DataFrame, output: Path) -> None:
    total_stations = len(station)
    lines = [
        "# 严格六年部署：年度站点覆盖口径",
        "",
        "## 1. 定义",
        "",
        "- 使用 `pwv_true`、`pwv_clim_surf_p1`、`pwv_gpt3` 同时为有限数值的共同 PWV 样本；",
        "- 某站某年 `common_pwv_N > 0` 时记为该年有可比样本；",
        "- 该覆盖指标仅描述既有严格部署结果文件中的可比样本可用性，不等同于原始 GNSS 观测完整率，也不表示所有站点每年均参与评估。",
        "",
        "## 2. 年度覆盖",
        "",
        "| year | 有可比样本的站点数 | 共同 PWV 样本数 |",
        "|---:|---:|---:|",
    ]
    for _, row in annual.iterrows():
        lines.append(f"| {int(row.year)} | {int(row.stations_with_samples)} | {int(row.common_pwv_N)} |")
    not_all_years = station.loc[station["years_with_samples"].lt(len(YEARS))]
    lines += [
        "",
        "## 3. 六年并集解释",
        "",
        f"- 六年并集为 {total_stations} 个站点；年度站点数可能小于并集站点数。",
        f"- 有至少一个年度缺少共同可比样本的站点数：{len(not_all_years)}。",
        "- 正式论文/汇报应明确区分“六年并集站点数”和“单年度可比站点数”，避免将并集站点数误写为每年稳定覆盖数量。",
        "",
        "## 4. 可复核文件",
        "",
        "- `annual_station_coverage_strict.csv`：按年汇总；",
        "- `station_coverage_summary_strict.csv`：按站汇总；",
        "- `station_year_coverage_matrix_strict.csv`：完整站点×年份矩阵。",
    ]
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="统计严格部署结果的年度站点覆盖")
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
    annual, station, matrix = build_coverage(load_common_samples(result_root))
    annual.to_csv(out_dir / "annual_station_coverage_strict.csv", index=False, encoding="utf-8-sig")
    station.to_csv(out_dir / "station_coverage_summary_strict.csv", index=False, encoding="utf-8-sig")
    matrix.to_csv(out_dir / "station_year_coverage_matrix_strict.csv", index=False, encoding="utf-8-sig")
    write_report(annual, station, out_dir / "年度站点覆盖口径说明_20260817.md")
    print(f"years={len(annual)}")
    print(f"union_stations={len(station)}")
    print(f"common_samples={int(annual['common_pwv_N'].sum())}")
    print(f"output={out_dir}")


if __name__ == "__main__":
    main()
