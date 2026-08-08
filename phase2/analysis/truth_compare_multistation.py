# -*- coding: utf-8 -*-
"""
多站真值对比: GNSS 反演 PWV (格网Pi / GPT3基线) vs 共位探空真值 (2019)
配对: wuh2-57494, urum-51463, bjfs-54511, shao-58362
"""
import os, sys
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.dirname(os.path.abspath(__file__))
BASE = r'D:\gnss水汽反演'
sys.path.insert(0, os.path.join(BASE, 'gpt3_1'))
from gpt3 import GPT3
gpt3 = GPT3(os.path.join(BASE, 'gpt3_1', 'gpt3_1.grd'))
K2P=22.1; K3=3.739e5; RV=461.495; RHO_W=1000.0
def pi_tm(Tm): return 1e8/(RHO_W*RV*(K3/Tm+K2P))

PAIRS = [
    ('wuh2', 'CHM00057494'),
    ('urum', 'CHM00051463'),
    ('bjfs', 'CHM00054511'),
    ('shao', 'CHM00058362'),
]
COORD = {}
with open(os.path.join(BASE, 'IGSwhu_formatted(1).txt'), encoding='utf-8') as fh:
    for ln in fh:
        ln = ln.strip()
        if not ln or ',' not in ln: continue
        p = [x.strip() for x in ln.split(',')]
        if len(p) >= 4:
            try: COORD[p[0][:4].lower()] = (float(p[1]), float(p[2]), float(p[3]))
            except Exception: pass


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
    r = df.loc[idx, ['TIME', 'PWV']].copy()
    r = r[r['TIME'].dt.year == 2019]
    r['date'] = r['TIME'].dt.date
    r['hour'] = r['TIME'].dt.hour
    return r


def metric(p, t):
    p = np.asarray(p, dtype=float); t = np.asarray(t, dtype=float)
    return {'RMSE': float(np.sqrt(mean_squared_error(t, p))),
            'MAE': float(mean_absolute_error(t, p)),
            'R2': float(r2_score(t, p)),
            'Bias': float(np.mean(p - t)), 'N': len(p)}


all_rows = []
for st, rad_id in PAIRS:
    gf = os.path.join(OUT, f'gnss_pwv_{st}.csv')
    rf = os.path.join(OUT, 'radiosonde_2019', f'{rad_id}_met.txt')
    if not os.path.exists(gf) or not os.path.exists(rf):
        print(f'{st}: 缺文件'); continue
    df = pd.read_csv(gf, parse_dates=['time'])
    df['date'] = df['time'].dt.date
    df['hour'] = df['time'].dt.hour
    df['doy'] = df['time'].dt.dayofyear
    rad = read_rad(rf)
    lat, lon, h = COORD[st]

    rows = []
    for hr in [0, 12]:
        gh = df[df['hour'] == hr]
        rh = rad[rad['hour'] == hr]
        rhd = rh.set_index('date')['PWV']
        for date, grp in gh.groupby('date'):
            if date not in rhd.index:
                continue
            zwd = grp['zwd_mm'].mean()
            pwv_ours = grp['pwv_mm'].mean()
            _, _, Tm, _ = gpt3.compute(lat, lon, h, 2019, int(grp['doy'].iloc[0]), float(hr))
            pwv_gpt3 = float(pi_tm(float(np.asarray(Tm).item()))) * zwd
            rows.append({'station': st, 'date': date, 'hour': hr, 'pwv_rad': float(rhd[date]),
                         'pwv_ours': pwv_ours, 'pwv_gpt3': pwv_gpt3})
    m = pd.DataFrame(rows)
    if len(m) == 0:
        print(f'{st}: 无匹配'); continue
    all_rows.append(m)
    mo = metric(m['pwv_ours'], m['pwv_rad'])
    mg = metric(m['pwv_gpt3'], m['pwv_rad'])
    print(f'{st:6s} ({rad_id}): N={mo["N"]:4d}  Ours RMSE={mo["RMSE"]:.2f} R2={mo["R2"]:.3f}  |  '
          f'GPT3 RMSE={mg["RMSE"]:.2f} R2={mg["R2"]:.3f}  |  Ours优势={100*(mg["RMSE"]-mo["RMSE"])/mg["RMSE"]:+.1f}%')

allm = pd.concat(all_rows, ignore_index=True)
print('\n===== 汇总 (4站合并) =====')
for hr in ['ALL', 0, 12]:
    sub = allm if hr == 'ALL' else allm[allm['hour'] == hr]
    mo = metric(sub['pwv_ours'], sub['pwv_rad'])
    mg = metric(sub['pwv_gpt3'], sub['pwv_rad'])
    impr = 100 * (mg['RMSE'] - mo['RMSE']) / mg['RMSE']
    print(f'{str(hr):4s} N={mo["N"]:4d}  Ours RMSE={mo["RMSE"]:.2f} R2={mo["R2"]:.3f}  |  '
          f'GPT3 RMSE={mg["RMSE"]:.2f} R2={mg["R2"]:.3f}  |  Ours优势={impr:+.1f}%')
allm.to_csv(os.path.join(OUT, 'truth_compare_multistation.csv'), index=False, float_format='%.3f')
print('\n已保存 truth_compare_multistation.csv')


if __name__ == '__main__':
    pass
