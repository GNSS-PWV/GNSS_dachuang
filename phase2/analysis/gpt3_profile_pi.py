# -*- coding: utf-8 -*-
"""
GPT3 背景廓线版 Pi (管线升级): 用 GPT3 构造垂直廓线喂给 ProfileTransformer 得到 Pi
对比: ①月尺度格网Pi(基于探空气候态) ②GPT3背景廓线Pi(纯经验模型) ③探空真值(武汉)
"""
import os, sys
import numpy as np
import pandas as pd
import pickle
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
OUT = os.path.dirname(os.path.abspath(__file__))
BASE = r'D:\gnss水汽反演'

GRD = os.path.join(BASE, 'gpt3_1', 'gpt3_1.grd')
MODEL_PTH = os.path.join(OUT, 'result_aligned_official', 'best_model.pth')
SCALERS = os.path.join(OUT, 'result_aligned_official', 'scalers.pkl')
GRID_CSV = os.path.join(OUT, 'result_grid_monthly', 'pi_grid_all.csv')
STATIONS = ['wuh2', 'hksl', 'hkws', 'shao', 'kmnm', 'bjfs', 'lhaz', 'urum', 'chan']
MONTH_DOY = {1: 15, 2: 45, 3: 74, 4: 105, 5: 135, 6: 166, 7: 196, 8: 227, 9: 258, 10: 288, 11: 319, 12: 349}


def encode_global(zwd, lat, lon, doy, hour):
    return np.array([zwd,
                     np.sin(np.deg2rad(lat)), np.cos(np.deg2rad(lat)),
                     np.sin(np.deg2rad(lon)), np.cos(np.deg2rad(lon)),
                     np.sin(2*np.pi*doy/365.0), np.cos(2*np.pi*doy/365.0),
                     np.sin(2*np.pi*hour/24.0), np.cos(2*np.pi*hour/24.0)], dtype=np.float32)


def metrics(p, t):
    p = np.asarray(p); t = np.asarray(t)
    return {'RMSE': float(np.sqrt(mean_squared_error(t, p))),
            'MAE': float(mean_absolute_error(t, p)),
            'R2': float(r2_score(t, p)),
            'Bias': float(np.mean(p - t)), 'N': len(p)}


def main():
    gpt3_dir = os.path.dirname(os.path.abspath(GRD))
    if gpt3_dir not in sys.path:
        sys.path.insert(0, gpt3_dir)
    from gpt3 import GPT3
    gpt3 = GPT3(GRD)

    from model import ProfileTransformer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt = torch.load(MODEL_PTH, map_location=device, weights_only=True)
    model = ProfileTransformer(
        d_model=ckpt.get('args', {}).get('d_model', 128),
        n_heads=ckpt.get('args', {}).get('n_heads', 8),
        n_layers=ckpt.get('args', {}).get('n_layers', 4),
        ff_dim=ckpt.get('args', {}).get('ff_dim', 512),
        dropout=ckpt.get('args', {}).get('dropout', 0.1),
        global_feat_dim=9).to(device)
    model.load_state_dict(ckpt['model_state_dict']); model.eval()
    with open(SCALERS, 'rb') as f:
        sc = pickle.load(f)
    ls = sc['level_scaler']; gs = sc['global_scaler']
    h_mean = sc['height_mean']; h_std = sc['height_std']
    print(f'模型: {MODEL_PTH}', flush=True)

    # 站点坐标
    coord = {}
    with open(os.path.join(BASE, 'IGSwhu_formatted(1).txt'), encoding='utf-8') as fh:
        for ln in fh:
            ln = ln.strip()
            if not ln or ',' not in ln: continue
            p = [x.strip() for x in ln.split(',')]
            if len(p) >= 4:
                try:
                    coord[p[0][:4].lower()] = (float(p[1]), float(p[2]), float(p[3]))
                except Exception:
                    pass

    # 月尺度格网 Pi (对照)
    grid = pd.read_csv(GRID_CSV)
    grid['month'] = grid['season'].str.extract(r'(\d+)').astype(int)

    rows = []
    for st in STATIONS:
        f = os.path.join(OUT, f'gnss_pwv_{st}.csv')
        if not os.path.exists(f):
            continue
        df = pd.read_csv(f, parse_dates=['time'])
        df['month'] = df['time'].dt.month
        lat, lon, h = coord[st]
        zwd_ref = float(df['zwd_mm'].mean())
        # 每月的 GPT3 廓线 Pi
        pi_g3 = {}
        with torch.no_grad():
            for m, doy in MONTH_DOY.items():
                heights = np.arange(max(0.0, h), 30000 + 1, 1000.0).astype(np.float32)
                p, T_c, Tm, e = gpt3.compute(np.full(len(heights), lat), np.full(len(heights), lon),
                                             heights, np.full(len(heights), 2019),
                                             np.full(len(heights), doy), np.full(len(heights), 12))
                levels = np.stack([heights, np.asarray(T_c) + 273.15, np.asarray(p), np.asarray(e)],
                                  axis=1).astype(np.float32)
                levels_n = ls.transform(levels).astype(np.float32)
                heights_n = ((levels[:, 0] - h_mean) / h_std).astype(np.float32)
                gf = gs.transform(encode_global(zwd_ref, lat, lon, doy, 12).reshape(1, -1))[0].astype(np.float32)
                L = len(levels)
                pi = model(torch.from_numpy(levels_n).unsqueeze(0).to(device),
                           torch.from_numpy(heights_n).unsqueeze(0).to(device),
                           torch.from_numpy(gf).unsqueeze(0).to(device),
                           torch.ones(1, L, dtype=torch.bool, device=device)).item()
                pi_g3[m] = pi
        df['pi_gpt3'] = df['month'].map(pi_g3)
        # 格网 Pi
        pi_grid_by_month = {}
        for m in range(1, 13):
            gm = grid[grid['month'] == m]
            d2 = (gm['lat'] - lat)**2 + (((gm['lon'] - lon + 180) % 360 - 180))**2
            pi_grid_by_month[m] = gm.loc[d2.idxmin(), 'pi']
        df['pi_grid'] = df['month'].map(pi_grid_by_month)
        df['pwv_gpt3'] = df['pi_gpt3'] * df['zwd_mm']
        df['pwv_grid'] = df['pi_grid'] * df['zwd_mm']
        df.to_csv(os.path.join(OUT, f'gnss_pwv_{st}_gpt3profile.csv'), index=False, float_format='%.3f')
        print(f'{st}: Pi_gpt3={pi_g3[1]:.4f}..{pi_g3[7]:.4f} (Jan..Jul)  Pi_grid={pi_grid_by_month[1]:.4f}..{pi_grid_by_month[7]:.4f}', flush=True)
        rows.append(df[['time', 'zwd_mm', 'pwv_mm', 'pwv_grid', 'pwv_gpt3']].assign(station=st))

    all_df = pd.concat(rows, ignore_index=True)
    all_df.to_csv(os.path.join(OUT, 'gnss_pwv_gpt3profile_all.csv'), index=False, float_format='%.3f')
    # 汇总对比 (以原 pwv_mm=格网版为基准, 比较 gpt3 版)
    print('\n=== Pi 来源对比 (GNSS PWV) ===')
    for st, g in all_df.groupby('station'):
        m_grid = metrics(g['pwv_grid'].values, g['pwv_mm'].values)
        m_g3 = metrics(g['pwv_gpt3'].values, g['pwv_mm'].values)
        print(f'{st:6s}: 格网Pi vs 原版 RMSE={m_grid["RMSE"]:.3f}  |  GPT3廓线Pi vs 原版 RMSE={m_g3["RMSE"]:.3f} R2={m_g3["R2"]:.3f}')

    # 武汉 vs 探空真值
    rad_f = os.path.join(OUT, 'radiosonde_2019', 'CHM00057494_met.txt')
    if os.path.exists(rad_f):
        rad = pd.read_csv(rad_f, header=0, sep=None, engine='python')
        fc = rad.columns[0]
        if str(fc).startswith('Unnamed') or str(fc).strip() == '':
            rad = rad.rename(columns={fc: 'TIME'})
        rad['TIME'] = pd.to_datetime(rad['TIME'], errors='coerce')
        rad['ELV'] = pd.to_numeric(rad['ELV'], errors='coerce')
        rad['PWV'] = pd.to_numeric(rad['PWV'], errors='coerce')
        rad = rad.dropna(subset=['ELV', 'PWV', 'TIME'])
        idx = rad.groupby('TIME')['ELV'].idxmin()
        rads = rad.loc[idx, ['TIME', 'PWV']].rename(columns={'TIME': 'time', 'PWV': 'pwv_rad'})
        rads = rads[rads.time.dt.year == 2019]
        g = all_df[all_df.station == 'wuh2'].copy()
        g['date'] = g['time'].dt.date; g['hour'] = g['time'].dt.hour
        rads['date'] = rads['time'].dt.date; rads['hour'] = rads['time'].dt.hour
        for hour in [0, 12]:
            gh = g[g.hour == hour].groupby('date')['pwv_mm'].mean().rename('pwv_grid')
            gh3 = g[g.hour == hour].groupby('date')['pwv_gpt3'].mean()
            rh = rads[rads.hour == hour].set_index('date')['pwv_rad']
            common = gh.index.intersection(rh.index)
            if len(common) >= 3:
                m1 = metrics(gh[common].values, rh[common].values)
                m2 = metrics(gh3[common].values, rh[common].values)
                print(f'\n武汉 {hour:02d}Z vs 探空: 格网Pi RMSE={m1["RMSE"]:.2f} R2={m1["R2"]:.3f}  |  GPT3廓线Pi RMSE={m2["RMSE"]:.2f} R2={m2["R2"]:.3f} (n={len(common)})')


if __name__ == '__main__':
    main()
