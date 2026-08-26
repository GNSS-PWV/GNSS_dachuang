# -*- coding: utf-8 -*-
"""
误差订正分析: 官方36站测试集

分析模型偏差结构, 量化不同订正策略的效果:
  1) 原始误差
  2) 全局偏差订正 (减去整体 bias)
  3) 逐站偏差订正 (减去各站历史 bias, 部署时可用历史数据校准 -> 上限参考)
  4) 偏差与 纬度/高程/PWV 的关系 (找系统性来源)

输入: test_predictions.csv (含 station_id/lat/lon/elv/pwv_true/pwv_pred)
输出: result_bias/bias_analysis.csv, bias_vs_lat.png, bias_vs_pwv.png, per_station_bias.csv
"""
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def metrics(pred, true):
    pred = np.asarray(pred); true = np.asarray(true)
    rmse = float(np.sqrt(mean_squared_error(true, pred)))
    mae = float(mean_absolute_error(true, pred))
    r2 = float(r2_score(true, pred))
    bias = float(np.mean(pred - true))
    return {'RMSE': rmse, 'MAE': mae, 'R2': r2, 'Bias': bias, 'N': len(true)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', required=True)
    ap.add_argument('--out', default='result_bias')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    df = pd.read_csv(args.csv)
    print(f'加载: N={len(df)}, 站数={df["station_id"].nunique()}', flush=True)

    err = df['pwv_pred'].values - df['pwv_true'].values
    m_raw = metrics(df['pwv_pred'].values, df['pwv_true'].values)
    # 全局订正
    gb = err.mean()
    m_global = metrics(df['pwv_pred'].values - gb, df['pwv_true'].values)
    # 逐站订正 (留一法: 用该站其他样本估计 bias, 避免泄漏)
    st_bias = df.groupby('station_id')['pwv_error'].transform('mean')
    n_per = df.groupby('station_id')['pwv_error'].transform('count')
    # 留一: 站内样本数>1 时用 (sum-b)/ (n-1)
    st_sum = df.groupby('station_id')['pwv_error'].transform('sum')
    loo_bias = (st_sum - err) / (n_per - 1)
    loo_bias = loo_bias.replace([np.inf, -np.inf], np.nan).fillna(0)
    m_loo = metrics(df['pwv_pred'].values - loo_bias, df['pwv_true'].values)

    print('\n=== 订正效果 ===')
    for name, m in [('原始', m_raw), ('全局订正', m_global), ('逐站订正(留一)', m_loo)]:
        print(f'{name:12s}: RMSE={m["RMSE"]:.4f}  MAE={m["MAE"]:.4f}  Bias={m["Bias"]:.4f}  R2={m["R2"]:.6f}')

    # 逐站偏差表
    per = df.groupby('station_id').agg(
        lat=('lat', 'mean'), lon=('lon', 'mean'), elv=('elv', 'mean'),
        N=('pwv_error', 'count'), bias=('pwv_error', 'mean'),
        rmse=('pwv_error', lambda s: (s ** 2).mean() ** 0.5),
        pwv_mean=('pwv_true', 'mean'))
    per.to_csv(os.path.join(args.out, 'per_station_bias.csv'), index=True, float_format='%.4f')
    print(f'\n逐站偏差: 最大正偏={per.bias.max():+.4f} ({per.bias.idxmax()}), 最大负偏={per.bias.min():+.4f} ({per.bias.idxmin()})')

    # 偏差 vs 纬度
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    axes[0].scatter(per['lat'], per['bias'], c='steelblue', s=30)
    axes[0].axhline(0, color='r', ls='--', lw=0.8)
    axes[0].set_xlabel('Latitude'); axes[0].set_ylabel('Per-station Bias (mm)')
    axes[0].set_title('Bias vs Latitude'); axes[0].grid(alpha=0.3)
    axes[1].scatter(per['elv'], per['bias'], c='darkorange', s=30)
    axes[1].axhline(0, color='r', ls='--', lw=0.8)
    axes[1].set_xlabel('Station Elevation (m)'); axes[1].set_ylabel('Bias (mm)')
    axes[1].set_title('Bias vs Elevation'); axes[1].grid(alpha=0.3)
    axes[2].scatter(per['pwv_mean'], per['bias'], c='green', s=30)
    axes[2].axhline(0, color='r', ls='--', lw=0.8)
    axes[2].set_xlabel('Station mean PWV (mm)'); axes[2].set_ylabel('Bias (mm)')
    axes[2].set_title('Bias vs PWV'); axes[2].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(args.out, 'bias_vs_factors.png'), dpi=200)
    plt.close()

    # 偏差随 PWV 分箱
    bins = [0, 10, 20, 30, 40, 50, 60, 80, 100]
    df['pwv_bin'] = pd.cut(df['pwv_true'], bins)
    bv = df.groupby('pwv_bin', observed=True)['pwv_error'].agg(['mean', 'count'])
    print('\n=== 偏差随 PWV 分箱 ===')
    print(bv.to_string())

    pd.DataFrame([{'method': 'raw', **m_raw}, {'method': 'global_correct', **m_global},
                  {'method': 'station_loo_correct', **m_loo}]).to_csv(
        os.path.join(args.out, 'bias_analysis.csv'), index=False, float_format='%.6f')
    print(f'\n完成 -> {args.out}')


if __name__ == '__main__':
    main()
