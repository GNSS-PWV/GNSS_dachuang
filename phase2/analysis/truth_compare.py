# -*- coding: utf-8 -*-
"""
真值对比: GNSS 反演 PWV (IGS ZTD + 格网Pi + GPT3) vs 探空 PWV (IGRA)
武汉站: IGS wuh2 vs 探空 CHM00057494 (2019, 00/12 UTC 匹配)
"""
import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

BASE = r'D:\gnss水汽反演'
OUT = os.path.join(BASE, '第二阶段')
RAD = os.path.join(OUT, 'radiosonde_2019', 'CHM00057494_met.txt')
GNSS = os.path.join(OUT, 'gnss_pwv_wuh2.csv')


def read_radiosonde(path):
    df = pd.read_csv(path, header=0, sep=None, engine='python')
    fc = df.columns[0]
    if str(fc).startswith('Unnamed') or str(fc).strip() == '':
        df = df.rename(columns={fc: 'TIME'})
    df['TIME'] = pd.to_datetime(df['TIME'], errors='coerce')
    for c in ['ELV', 'PWV']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df = df.dropna(subset=['ELV', 'PWV', 'TIME'])
    # 地表层 = 最小 ELV
    idx = df.groupby('TIME')['ELV'].idxmin()
    surf = df.loc[idx, ['TIME', 'PWV']].copy()
    surf = surf.sort_values('TIME').reset_index(drop=True)
    return surf


def metrics(pred, true):
    pred = np.asarray(pred); true = np.asarray(true)
    return {'RMSE': float(np.sqrt(mean_squared_error(true, pred))),
            'MAE': float(mean_absolute_error(true, pred)),
            'R2': float(r2_score(true, pred)),
            'Bias': float(np.mean(pred - true)),
            'N': len(true)}


def main():
    rad = read_radiosonde(RAD)
    rad = rad[rad['TIME'].dt.year == 2019]
    rad = rad.rename(columns={'TIME': 'time', 'PWV': 'pwv_rad'})
    print(f'探空 2019: {len(rad)} 条 (00/12 UTC)', flush=True)

    g = pd.read_csv(GNSS, parse_dates=['time'])
    # 只取 00:00 和 12:00 UTC 附近的 GNSS PWV (探空时刻)
    g = g.set_index('time')
    g00 = g[g.index.hour == 0].resample('D').mean().reset_index()
    g12 = g[g.index.hour == 12].resample('D').mean().reset_index()
    gsel = pd.concat([g00, g12]).sort_values('time')
    gsel = gsel.rename(columns={'pwv_mm': 'pwv_gnss'})

    # 匹配探空
    rad['date'] = rad['time'].dt.date
    rad['hour'] = rad['time'].dt.hour
    gsel['date'] = gsel['time'].dt.date
    gsel['hour'] = gsel['time'].dt.hour
    m = gsel.merge(rad[['date', 'hour', 'pwv_rad']], on=['date', 'hour'], how='inner')
    m = m.dropna(subset=['pwv_gnss', 'pwv_rad'])
    print(f'匹配样本: {len(m)}', flush=True)

    mm = metrics(m['pwv_gnss'].values, m['pwv_rad'].values)
    print('\n===== GNSS vs 探空 PWV (武汉 2019) =====')
    print(f'  RMSE={mm["RMSE"]:.3f} mm   MAE={mm["MAE"]:.3f}   R2={mm["R2"]:.4f}   Bias={mm["Bias"]:.3f}   N={mm["N"]}')
    m[['date', 'hour', 'pwv_rad', 'pwv_gnss']].to_csv(
        os.path.join(OUT, 'truth_compare_wuhan.csv'), index=False, float_format='%.3f')

    # 散点图
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    ax = axes[0]
    ax.scatter(m['pwv_rad'], m['pwv_gnss'], s=12, alpha=0.5, color='steelblue')
    v0, v1 = 0, max(m['pwv_rad'].max(), m['pwv_gnss'].max()) * 1.05
    ax.plot([v0, v1], [v0, v1], 'r--', lw=1)
    ax.set_xlabel('Radiosonde PWV (mm)'); ax.set_ylabel('GNSS PWV (mm)')
    ax.set_title(f'GNSS vs Radiosonde PWV (Wuhan 2019)\nRMSE={mm["RMSE"]:.2f} R2={mm["R2"]:.4f}')
    ax.grid(alpha=0.3)
    ax = axes[1]
    ax.plot(m['date'], m['pwv_rad'], 'k-', lw=0.8, label='Radiosonde')
    ax.plot(m['date'], m['pwv_gnss'], 'b-', lw=0.8, alpha=0.8, label='GNSS (this method)')
    ax.set_xlabel('Date (2019)'); ax.set_ylabel('PWV (mm)')
    ax.set_title('Wuhan PWV time series')
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, 'truth_compare_wuhan.png'), dpi=150)
    print(f'\n完成: truth_compare_wuhan.csv + truth_compare_wuhan.png')


if __name__ == '__main__':
    main()
