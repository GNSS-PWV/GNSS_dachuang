# -*- coding: utf-8 -*-
"""
Saastamoinen 传统模型批量评估脚本
- 读取站点探空数据文件, 提取地表层 (最低ELV) 的 P/T/e/Tm/lat/h
- 用 Saastamoinen 模型计算 PWV
- 与数据中的真值 PWV 对比, 输出 RMSE/MAE/R2/Bias 及散点图

用法:
  python eval_saastamoinen.py [--data_dir PATH] [--output_dir PATH]

默认数据目录: 第一阶段/xg_test (本地测试集)
"""
import os
import sys
import glob
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 同目录导入
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from saastamoinen_pwv import saastamoinen_pwv_batch


COLUMN_NAMES = ["TIME", "YEAR", "DOY", "LAT", "LON", "ELV",
                "TS", "PS", "WPS", "ZWD", "ZHD", "ZTD", "PWV", "Tm"]


def read_surface_data(file_path):
    """
    读取一个站点文件, 每个时间戳取地表层 (最低ELV) 的一行.
    返回 DataFrame: LAT, LON, ELV, TS, PS, WPS, ZWD, ZHD, ZTD, PWV, Tm, DOY, YEAR, station_id
    """
    station_id = os.path.splitext(os.path.basename(file_path))[0].split("_")[0]
    try:
        df = pd.read_csv(file_path, header=0, sep=None, engine="python")
    except Exception:
        df = pd.read_csv(file_path, header=None, names=COLUMN_NAMES, sep=None, engine="python")
        return pd.DataFrame()

    first_col = df.columns[0]
    if str(first_col).startswith("Unnamed") or str(first_col).strip() == "":
        df = df.rename(columns={first_col: "TIME"})

    cols = [c for c in COLUMN_NAMES if c in df.columns]
    df = df[cols].copy()

    for c in ["ELV", "TS", "PS", "WPS", "ZWD", "PWV", "Tm", "LAT", "DOY", "YEAR"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["ELV", "TS", "PS", "WPS", "PWV", "Tm", "LAT"])

    # 每个时间戳取最低ELV行 (地表层)
    if "TIME" in df.columns:
        df["TIME"] = pd.to_datetime(df["TIME"], errors="coerce")
        df = df.sort_values(["TIME", "ELV"])
        surface = df.groupby("TIME", sort=True).first().reset_index()
    else:
        df = df.sort_values(["YEAR", "DOY", "ELV"])
        group_cols = [c for c in ["YEAR", "DOY"] if c in df.columns]
        if group_cols:
            surface = df.groupby(group_cols, sort=True).first().reset_index()
        else:
            surface = df.iloc[[0]].copy()

    surface["station_id"] = station_id
    return surface


def compute_metrics(pred, true):
    pred = np.asarray(pred, dtype=np.float64)
    true = np.asarray(true, dtype=np.float64)
    rmse = np.sqrt(mean_squared_error(true, pred))
    mae = mean_absolute_error(true, pred)
    r2 = r2_score(true, pred)
    bias = np.mean(pred - true)
    # 相对误差
    rel_rmse = rmse / np.mean(true) * 100
    return {
        "RMSE": rmse, "MAE": mae, "R2": r2, "Bias": bias,
        "Rel_RMSE_pct": rel_rmse, "N": len(true),
    }


def plot_scatter(true, pred, save_path, title="Saastamoinen PWV: Predicted vs True"):
    true = np.asarray(true).flatten()
    pred = np.asarray(pred).flatten()
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(true, pred, s=4, alpha=0.25, c="steelblue")
    vmin = float(min(true.min(), pred.min()))
    vmax = float(max(true.max(), pred.max()))
    ax.plot([vmin, vmax], [vmin, vmax], "r--", linewidth=1)
    ax.set_xlabel("True PWV (mm)")
    ax.set_ylabel("Predicted PWV (mm)")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Saastamoinen PWV 批量评估")
    parser.add_argument("--data_dir", default=r"D:\gnss水汽反演\第一阶段\xg_test",
                        help="站点数据文件目录")
    parser.add_argument("--output_dir", default=r"D:\gnss水汽反演\saastamoinen_result",
                        help="输出目录")
    args = parser.parse_args()

    data_dir = args.data_dir
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # 查找数据文件
    file_list = sorted(glob.glob(os.path.join(data_dir, "**", "*.txt"), recursive=True))
    if not file_list:
        file_list = sorted(glob.glob(os.path.join(data_dir, "*.txt")))
    if not file_list:
        print(f"错误: 未在 {data_dir} 找到数据文件!")
        return
    print(f"找到 {len(file_list)} 个站点文件", flush=True)

    # 读取所有站点的地表层数据
    all_surface = []
    for fp in file_list:
        df = read_surface_data(fp)
        if len(df) > 0:
            all_surface.append(df)
    if not all_surface:
        print("无有效数据!")
        return
    data = pd.concat(all_surface, ignore_index=True)
    print(f"共 {len(data)} 条地表层样本, 来自 {data['station_id'].nunique()} 个站点", flush=True)

    # 提取输入
    lat = data["LAT"].values
    h = data["ELV"].values
    P = data["PS"].values
    e = data["WPS"].values
    T = data["TS"].values       # 开尔文
    Tm = data["Tm"].values       # 开尔文
    pwv_true = data["PWV"].values

    # Saastamoinen 计算 PWV
    print("正在用 Saastamoinen 模型计算 PWV...", flush=True)
    pwv_pred = saastamoinen_pwv_batch(lat, h, P, e, T, Tm)

    # 评估
    m = compute_metrics(pwv_pred, pwv_true)
    print("=" * 60, flush=True)
    print(f"Saastamoinen PWV 评估结果 (真值输入)", flush=True)
    print(f"  样本数:     {m['N']}", flush=True)
    print(f"  RMSE:       {m['RMSE']:.4f} mm", flush=True)
    print(f"  MAE:        {m['MAE']:.4f} mm", flush=True)
    print(f"  R2:         {m['R2']:.4f}", flush=True)
    print(f"  Bias:       {m['Bias']:.4f} mm", flush=True)
    print(f"  Rel RMSE:   {m['Rel_RMSE_pct']:.2f}%", flush=True)
    print("=" * 60, flush=True)

    # 同时评估 ZWD (Saastamoinen 算出的 ZWD vs 数据中的 ZWD)
    lat_rad = np.deg2rad(lat)
    f = 1.0 - 0.00266 * np.cos(2.0 * lat_rad) - 0.00000028 * h
    ztd_pred = 0.002277 * (P / f + (0.05 + 1255.0 / T) * e) * 1000.0
    zhd_pred = 0.002277 * P / f * 1000.0
    zwd_pred = ztd_pred - zhd_pred
    if "ZWD" in data.columns:
        zwd_true = data["ZWD"].values
        zm = compute_metrics(zwd_pred, zwd_true)
        print(f"  ZWD RMSE:   {zm['RMSE']:.4f} mm  R2={zm['R2']:.4f}  Bias={zm['Bias']:.4f}", flush=True)
        print("=" * 60, flush=True)

    # 保存结果
    result_df = data[["station_id", "DOY", "YEAR", "LAT", "ELV", "PS", "TS", "WPS", "Tm", "PWV"]].copy()
    result_df["PWV_pred"] = pwv_pred
    result_df["PWV_error"] = pwv_pred - pwv_true
    result_df["ZWD_pred"] = zwd_pred
    result_csv = os.path.join(output_dir, "saastamoinen_pwv_results.csv")
    result_df.to_csv(result_csv, index=False, float_format="%.6f")
    print(f"结果已保存: {result_csv}", flush=True)

    # 保存指标
    with open(os.path.join(output_dir, "metrics.txt"), "w", encoding="utf-8") as f:
        f.write("Saastamoinen PWV 评估结果 (真值输入)\n")
        f.write("=" * 60 + "\n")
        f.write(f"样本数:     {m['N']}\n")
        f.write(f"RMSE:       {m['RMSE']:.4f} mm\n")
        f.write(f"MAE:        {m['MAE']:.4f} mm\n")
        f.write(f"R2:         {m['R2']:.4f}\n")
        f.write(f"Bias:       {m['Bias']:.4f} mm\n")
        f.write(f"Rel RMSE:   {m['Rel_RMSE_pct']:.2f}%\n")
        if "ZWD" in data.columns:
            f.write(f"\nZWD RMSE:   {zm['RMSE']:.4f} mm\n")
            f.write(f"ZWD R2:     {zm['R2']:.4f}\n")
            f.write(f"ZWD Bias:   {zm['Bias']:.4f} mm\n")

    # 散点图
    plot_scatter(pwv_true, pwv_pred, os.path.join(output_dir, "scatter_pwv_saastamoinen.png"))
    print(f"散点图已保存: {os.path.join(output_dir, 'scatter_pwv_saastamoinen.png')}", flush=True)
    print("评估完成!", flush=True)


if __name__ == "__main__":
    main()
