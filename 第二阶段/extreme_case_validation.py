# -*- coding: utf-8 -*-
"""
极端个例验证: 从 GNSS PWV 时间序列中提取暴雨事件, 与共位探空对比
个例: 武汉 2019-06 暴雨 (wuh2 vs 探空 CHM00057494); 香港 2019-07 (hksl/hkws)
"""
import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score

OUT = os.path.dirname(os.path.abspath(__file__))
GNSS = {s: os.path.join(OUT, f'gnss_pwv_{s}.csv') for s in ['wuh2', 'hksl', 'hkws']}
RAD = os.path.join(OUT, 'radiosonde_2019', 'CHM00057494_met.txt')


def read_rad(path):
    df = pd.read_csv(path, header=0, sep=None, engine='python')
    fc = df.columns[0]
    if str(fc).startswith('Unnamed') or str(fc).strip() == '':
        df = df.rename(columns={fc: 'TIME'})
    df['TIME'] = pd.to_datetime(df['TIME'], errors='coerce')
    df['ELV'] = pd.to_numeric(df['ELV'], errors='coerce')
    df['PWV'] = pd.to_numeric(df['PWV'], errors='coerce')
    df = df.dropna(subset=['ELV', 'PWV', 'TIME'])
    idx = df.groupby('TIME')['ELV'].idxmin()
    return df.loc[idx, ['TIME', 'PWV']].rename(columns={'TIME': 'time', 'PWV': 'pwv_rad'})


def metrics(p, t):
    p = np.asarray(p); t = np.asarray(t)
    return float(np.sqrt(mean_squared_error(t, p))), float(r2_score(t, p)), len(p)


def main():
    rad = read_rad(RAD)
    rad = rad[(rad.time.dt.year == 2019)].sort_values('time')
    r0 = rad[rad.time.dt.hour == 0]
    rad00 = r0.set_index(r0.time.dt.date)['pwv_rad']
    r1 = rad[rad.time.dt.hour == 12]
    rad12 = r1.set_index(r1.time.dt.date)['pwv_rad']

    cases = [
        ('wuh2', 'Wuhan (GNSS wuh2)', '2019-06-10', '2019-06-30',
         rad00, rad12, 'Wuhan rainstorm Jun 2019'),
        ('hksl', 'Hong Kong (GNSS hksl)', '2019-06-25', '2019-07-10',
         None, None, 'Hong Kong rainstorm Jul 2019'),
    ]

    for st, label, t0, t1, r00, r12, title in cases:
        g = pd.read_csv(GNSS[st], parse_dates=['time'])
        g = g.set_index('time')
        # 小时平均
        gh = g['pwv_mm'].resample('1h').mean()
        seg = gh[(gh.index >= t0) & (gh.index <= t1)]
        if seg.empty:
            print(f'{st}: 无数据'); continue
        fig, ax = plt.subplots(figsize=(13, 4.5))
        ax.plot(seg.index, seg.values, lw=1.0, color='steelblue', label='GNSS PWV (this method)')
        if r00 is not None:
            d00 = r00[(r00.index >= pd.Timestamp(t0).date()) & (r00.index <= pd.Timestamp(t1).date())]
            d12 = r12[(r12.index >= pd.Timestamp(t0).date()) & (r12.index <= pd.Timestamp(t1).date())]
            ax.plot(pd.to_datetime(d00.index), d00.values, 'k.', ms=5, label='Radiosonde 00Z')
            ax.plot(pd.to_datetime(d12.index), d12.values, 'r.', ms=5, label='Radiosonde 12Z')
            # 匹配对比
            dates = d00.index.intersection(g[g.index.hour == 0].index.date)
            gg0 = g[g.index.hour == 0]
            g00 = gg0.groupby(gg0.index.date)['pwv_mm'].mean()
            common = d00.index.intersection(g00.index)
            if len(common) >= 3:
                rmse, r2, n = metrics(g00[common].values, d00[common].values)
                ax.set_title(f'{title}  |  GNSS vs Radiosonde(00Z): RMSE={rmse:.2f}mm R2={r2:.3f} (n={n})')
        ax.set_xlabel('Time (2019)'); ax.set_ylabel('PWV (mm)')
        ax.legend(); ax.grid(alpha=0.3)
        plt.tight_layout()
        fn = os.path.join(OUT, f'extreme_{st}.png')
        plt.savefig(fn, dpi=150); plt.close()
        peak = seg.idxmax()
        print(f'{st}: 峰值 PWV={seg.max():.1f}mm @ {peak} -> {fn}')

    # 武汉个例定量汇总(全匹配)
    g = pd.read_csv(GNSS['wuh2'], parse_dates=['time'])
    g = g.set_index('time')
    g0 = g[g.index.hour == 0]
    g00 = g0.groupby(g0.index.date)['pwv_mm'].mean()
    g1 = g[g.index.hour == 12]
    g12 = g1.groupby(g1.index.date)['pwv_mm'].mean()
    common = rad00.index.intersection(g00.index)
    rmse, r2, n = metrics(g00[common].values, rad00[common].values)
    print(f'\n武汉 全年 00Z: GNSS vs 探空 RMSE={rmse:.2f}mm R2={r2:.3f} n={n}')
    common = rad12.index.intersection(g12.index)
    rmse, r2, n = metrics(g12[common].values, rad12[common].values)
    print(f'武汉 全年 12Z: GNSS vs 探空 RMSE={rmse:.2f}mm R2={r2:.3f} n={n}')


if __name__ == '__main__':
    main()
