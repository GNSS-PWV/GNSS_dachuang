# -*- coding: utf-8 -*-
"""
第三阶段分析: 测试集分层误差评估 (季节 / 纬度带 / 水汽强度)

输入: test_predictions.csv (含 station_id/time/lat/lon/elv/zwd/pwv_true/pwv_pred/pi_true/pi_pred)
输出: result_phase3/
  - phase3_summary.csv     各分组指标汇总
  - phase3_bar_season.png  季节 RMSE/BIAS 柱状图
  - phase3_bar_lat.png     纬度带 RMSE/BIAS 柱状图
  - phase3_bar_zwd.png     ZWD 分箱 RMSE/BIAS 柱状图
  - phase3_error_vs_zwd.png 误差随 ZWD 变化散点
"""
import os
import argparse
import numpy as np
import pandas as pd
import datetime as dt
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def compute_metrics(pred, true):
    pred = np.asarray(pred, dtype=np.float64).flatten()
    true = np.asarray(true, dtype=np.float64).flatten()
    rmse = float(np.sqrt(mean_squared_error(true, pred)))
    mae = float(mean_absolute_error(true, pred))
    r2 = float(r2_score(true, pred))
    bias = float(np.mean(pred - true))
    rel = rmse / np.mean(true) * 100 if np.mean(true) > 0 else 0.0
    return {'RMSE': rmse, 'MAE': mae, 'R2': r2, 'Bias': bias, 'Rel_RMSE_pct': rel, 'N': len(true)}


def season_of(month):
    if month in (12, 1, 2):
        return 'DJF'
    if month in (3, 4, 5):
        return 'MAM'
    if month in (6, 7, 8):
        return 'JJA'
    return 'SON'


def lat_band_of(lat):
    a = abs(lat)
    if a >= 60:
        return 'Polar(60-90)'
    if a >= 30:
        return 'Mid(30-60)'
    if a >= 15:
        return 'SubTrop(15-30)'
    return 'Trop(0-15)'


def zwd_bin_of(z):
    if z < 50:
        return 'ZWD<50'
    if z < 100:
        return '50-100'
    if z < 200:
        return '100-200'
    return 'ZWD>200'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', required=True)
    ap.add_argument('--out', default='result_phase3')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    df = pd.read_csv(args.csv)
    print(f'加载: N={len(df)}', flush=True)

    # 时间解析
    months, seasons = [], []
    for s in df['time']:
        try:
            t = dt.datetime.strptime(str(s)[:19], '%Y-%m-%d %H:%M:%S')
        except Exception:
            t = pd.to_datetime(s)
        months.append(t.month)
        seasons.append(season_of(t.month))
    df['month'] = months
    df['season'] = seasons
    df['lat_band'] = df['lat'].apply(lat_band_of)
    df['zwd_bin'] = df['zwd'].apply(zwd_bin_of)
    df['abs_lat'] = df['lat'].abs()
    df['hemi'] = np.where(df['lat'] >= 0, 'North', 'South')

    groups = {
        'season': df['season'],
        'lat_band': df['lat_band'],
        'zwd_bin': df['zwd_bin'],
        'hemi': df['hemi'],
    }

    rows = []
    for gname, gseries in groups.items():
        for gval, sub in df.groupby(gseries):
            m = compute_metrics(sub['pwv_pred'].values, sub['pwv_true'].values)
            m['group'] = gname
            m['value'] = gval
            m['pwv_mean'] = float(sub['pwv_true'].mean())
            m['zwd_mean'] = float(sub['zwd'].mean())
            rows.append(m)
    summary = pd.DataFrame(rows)
    summary.to_csv(os.path.join(args.out, 'phase3_summary.csv'), index=False, float_format='%.6f')
    print(summary.to_string(index=False))
    print(f'\n已保存: {os.path.join(args.out, "phase3_summary.csv")}')

    # 柱状图
    def bar(key, fname, title):
        sub = summary[summary['group'] == key].sort_values('value')
        fig, ax1 = plt.subplots(figsize=(10, 4.5))
        x = np.arange(len(sub))
        ax1.bar(x, sub['RMSE'].values, color='steelblue', label='RMSE')
        ax1.set_ylabel('RMSE (mm)')
        ax1.set_xticks(x); ax1.set_xticklabels(sub['value'].values, rotation=20, ha='right')
        ax2 = ax1.twinx()
        ax2.plot(x, sub['Bias'].values, 'ro-', label='Bias')
        ax2.axhline(0, color='gray', lw=0.8)
        ax2.set_ylabel('Bias (mm)', color='red')
        for i, (r, n) in enumerate(zip(sub['RMSE'].values, sub['N'].values)):
            ax1.text(i, r + 0.002, f'{r:.3f}\n(n={n})', ha='center', fontsize=8)
        ax1.set_title(title)
        ax1.grid(alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(os.path.join(args.out, fname), dpi=200)
        plt.close()
        print(f'图已保存: {fname}')

    bar('season', 'phase3_bar_season.png', 'Seasonal PWV RMSE (ProfileTransformer)')
    bar('lat_band', 'phase3_bar_lat.png', 'Latitude-band PWV RMSE')
    bar('zwd_bin', 'phase3_bar_zwd.png', 'PWV RMSE by ZWD intensity')

    # 误差 vs ZWD 散点
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(df['zwd'].values, df['pwv_error'].values, s=3, alpha=0.15, color='steelblue')
    # 分箱均值
    bins = np.arange(0, 450, 25)
    idx = np.digitize(df['zwd'].values, bins)
    for b in np.unique(idx):
        sel = idx == b
        if sel.sum() >= 20:
            ax.plot(df['zwd'].values[sel].mean(), df['pwv_error'].values[sel].mean(), 'ro')
    ax.axhline(0, color='r', lw=0.8, ls='--')
    ax.set_xlabel('ZWD (mm)'); ax.set_ylabel('PWV Error (mm)')
    ax.set_title('Transformer PWV Error vs ZWD')
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(args.out, 'phase3_error_vs_zwd.png'), dpi=200)
    plt.close()

    print(f'\n全部完成 -> {args.out}')


if __name__ == '__main__':
    main()
