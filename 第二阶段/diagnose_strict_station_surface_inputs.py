"""对严格六年部署结果做一阶段地表输入误差的站点关联诊断。

仅从 result_p1deploy_ft_strict_20260817 的六个年度预测 CSV 读取数据；
不联网、不下载、不修改原始预测。诊断使用与 PWV 比较相同的共同样本队列。
Tm 仅作为诊断量，不是二阶段模型输入。
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

METHOD_COLUMNS = {
    "clim_surf_p1": "pwv_clim_surf_p1",
    "gpt3": "pwv_gpt3",
}
INPUT_COLUMNS = {
    "ps": ("ps_true", "ps_p1", "PS"),
    "ts": ("ts_true", "ts_p1", "TS"),
    "wps": ("wps_true", "wps_p1", "WPS"),
    "tm": ("tm_true", "tm_p1", "Tm（仅诊断，不是二阶段输入）"),
}


def error_metrics(frame: pd.DataFrame, truth_col: str, prediction_col: str) -> dict[str, float | int]:
    truth = pd.to_numeric(frame[truth_col], errors="coerce").to_numpy(dtype=float)
    prediction = pd.to_numeric(frame[prediction_col], errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(truth) & np.isfinite(prediction)
    err = prediction[valid] - truth[valid]
    if not err.size:
        return {"N": 0, "RMSE": float("nan"), "MAE": float("nan"), "Bias": float("nan")}
    return {
        "N": int(err.size),
        "RMSE": float(np.sqrt(np.mean(np.square(err)))),
        "MAE": float(np.mean(np.abs(err))),
        "Bias": float(np.mean(err)),
    }


def load_common_samples(result_root: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    required = [
        "station",
        "time",
        "pwv_true",
        *METHOD_COLUMNS.values(),
        *(column for pair in INPUT_COLUMNS.values() for column in pair[:2]),
    ]
    required_set = set(required)
    for year in range(2014, 2020):
        path = result_root / f"year_{year}" / f"deploy_p1_predictions_{year}.csv"
        if not path.is_file():
            raise FileNotFoundError(f"缺少年度预测文件：{path}")
        frame = pd.read_csv(path, usecols=lambda name: name in required_set)
        missing = required_set - set(frame.columns)
        if missing:
            raise ValueError(f"{path.name} 缺少列：{sorted(missing)}")
        frame["year"] = year
        frame["time"] = pd.to_datetime(frame["time"], errors="coerce")
        frames.append(frame)
    data = pd.concat(frames, ignore_index=True)
    numeric_cols = [
        "pwv_true",
        *METHOD_COLUMNS.values(),
        *(column for pair in INPUT_COLUMNS.values() for column in pair[:2]),
    ]
    for column in numeric_cols:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    pwv_cols = ["pwv_true", *METHOD_COLUMNS.values()]
    finite_pwv = np.isfinite(data[pwv_cols].to_numpy(dtype=float)).all(axis=1)
    data = data.loc[data["station"].notna() & data["time"].notna() & finite_pwv].copy()
    return data


def station_diagnostics(data: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for station, group in data.groupby("station", sort=True, observed=True):
        surf = error_metrics(group, "pwv_true", METHOD_COLUMNS["clim_surf_p1"])
        gpt3 = error_metrics(group, "pwv_true", METHOD_COLUMNS["gpt3"])
        improvement = gpt3["RMSE"] - surf["RMSE"]
        if surf["N"] != gpt3["N"]:
            raise RuntimeError(f"{station}: PWV common-sample count mismatch after finite-value filtering.")
        row: dict[str, object] = {
            "station": station,
            "pwv_common_N": surf["N"],
            "N_clim_surf_p1": surf["N"],
            "N_gpt3": gpt3["N"],
            "rmse_clim_surf_p1_mm": surf["RMSE"],
            "rmse_gpt3_mm": gpt3["RMSE"],
            "gpt3_minus_clim_surf_p1_rmse_mm": improvement,
            "surf_vs_gpt3": "WIN" if improvement > 0 else "LOSS" if improvement < 0 else "TIE",
        }
        for key, (truth_col, prediction_col, _) in INPUT_COLUMNS.items():
            values = error_metrics(group, truth_col, prediction_col)
            row[f"N_{key}"] = values["N"]
            row[f"rmse_{key}_p1"] = values["RMSE"]
            row[f"mae_{key}_p1"] = values["MAE"]
            row[f"bias_{key}_p1"] = values["Bias"]
        rows.append(row)
    return pd.DataFrame(rows).sort_values("gpt3_minus_clim_surf_p1_rmse_mm").reset_index(drop=True)


def write_report(summary: pd.DataFrame, output: Path) -> None:
    weak = summary.loc[summary["surf_vs_gpt3"].ne("WIN")].copy()
    lines = [
        "# 严格六年部署：弱站一阶段地表输入误差探索性关联诊断（非因果归因）",
        "",
        "## 1. 口径与边界",
        "",
        "- 本诊断基于提供的 2014–2019 年严格部署结果文件；脚本本身不验证上游训练截止时间、数据划分或预测生成过程。",
        "- PWV 比较使用 `pwv_true`、`pwv_clim_surf_p1` 和 `pwv_gpt3` 三者均为有限数值的同一共同样本；`N_clim_surf_p1` 与 `N_gpt3` 同时写入 CSV 以便复核。",
        "- 站点的弱/强划分仅按 `clim_surf_p1` 相对 GPT3 的同站 RMSE 比较确定；`GPT3 − clim_surf_p1 RMSE` 为正表示本模型更优。",
        "- PS、TS、WPS 的误差为一阶段预测相对于真值的误差；Tm 同时列出，但**仅用于诊断，不是二阶段模型输入**。各输入指标在共同 PWV 队列中按该输入真值与 P1 预测同时有效的子集计算，实际样本量见 `N_ps`、`N_ts`、`N_wps`、`N_tm`。",
        "- Bias = P1 − true；正值表示一阶段 P1 预测偏高，负值表示偏低。",
        "- 这是由当前评估期事后产生的探索性关联分析：它只能观察输入误差模式与最终 PWV 表现是否并存，不能单独证明任何输入、地形、气候或观测因素造成弱站表现。",
        "",
        "## 2. 当前评估期中未优于 GPT3 的探索性名单",
        "",
        "| station | PWV N | N_ps | N_ts | N_wps | N_tm | clim_surf_p1 RMSE (mm) | GPT3 RMSE (mm) | GPT3−clim_surf_p1 (mm) | PS P1 RMSE | TS P1 RMSE | WPS P1 RMSE | Tm P1 RMSE（仅诊断） |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in weak.iterrows():
        lines.append(
            f"| {row.station} | {int(row.pwv_common_N)} | {int(row.N_ps)} | {int(row.N_ts)} | "
            f"{int(row.N_wps)} | {int(row.N_tm)} | {row.rmse_clim_surf_p1_mm:.6f} | "
            f"{row.rmse_gpt3_mm:.6f} | {row.gpt3_minus_clim_surf_p1_rmse_mm:.6f} | "
            f"{row.rmse_ps_p1:.6f} | {row.rmse_ts_p1:.6f} | {row.rmse_wps_p1:.6f} | {row.rmse_tm_p1:.6f} |"
        )
    lines += [
        "",
        "## 3. 可复核文件",
        "",
        "- `station_surface_input_diagnostics.csv`：全部站点的共同 PWV 样本数、各输入有效子集样本数、P1 输入误差及对应 PWV 对比结果。",
        "- 本报告：仅摘列当前评估期中未优于 GPT3 的站点，作为探索性假设生成材料。",
        "",
        "## 4. 后续独立验证要求",
        "",
        "1. 可将这些站点与坐标、海拔、气候分区和观测覆盖率关联，但结论只能表述为假设或相关性；",
        "2. 在训练截止年之后的 ERA5/NWP 剖面驱动实验中，应预先固定当前名单与判定规则，同时报告全部预先定义的目标站点和总体结果，不能只报告名单内站点；",
        "3. 不在当前六年测试集按站点选择最优二阶段方案后重新报告总体精度，以避免测试集选择偏差。",
    ]
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
def main() -> None:
    parser = argparse.ArgumentParser(description="输出严格六年部署的站点一阶段输入误差关联诊断")
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
    summary = station_diagnostics(data)
    summary.to_csv(out_dir / "station_surface_input_diagnostics.csv", index=False, encoding="utf-8-sig")
    write_report(summary, out_dir / "弱站一阶段地表输入误差关联诊断_20260817.md")

    print(f"common_samples={len(data)}")
    print(f"stations={len(summary)}")
    print(f"weak_stations={(summary['surf_vs_gpt3'] != 'WIN').sum()}")
    print(f"output={out_dir}")


if __name__ == "__main__":
    main()

