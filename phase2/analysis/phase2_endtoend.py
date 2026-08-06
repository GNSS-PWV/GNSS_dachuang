# -*- coding: utf-8 -*-
"""
端到端两步法对比 (第一阶段 GRU-Tm -> 传统两步法 PWV) vs GPT3 vs ProfileTransformer
==================================================================================
同一批样本(2017年, 第一阶段 Tm GRU 预测覆盖的站)上比较:
  1) 传统两步法: PWV = Pi(Tm_GRU) * ZWD   (Tm 来自第一阶段 GRU)
  2) GPT3:       PWV = Pi(Tm_GPT3) * ZWD
  3) Transformer:PWV = Pi(廓线) * ZWD     (第二阶段模型)
真值: 探空地表 PWV

用法:
  python phase2_endtoend.py
"""
import os, sys, glob
import pickle
import numpy as np
import pandas as pd
import datetime as dt
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

BASE = '/share/home/u23114/tj23114/packages/dachuang_pwv'
DATA = f'{BASE}/PS/xg_data'
GRD = f'{BASE}/gpt3_1/gpt3_1.grd'
TM_PRED = f'{BASE}/Ts_Tm/tm_result/gru/predictions/Tm/year_2017/*.txt'
MODEL_DIR = 'result_aligned'

K2P = 22.1; K3 = 3.739e5; RV = 461.495; RHO_W = 1000.0


def pi_from_tm(Tm):
    return 1e8 / (RHO_W * RV * (K3 / Tm + K2P))


def metrics(pred, true):
    pred = np.asarray(pred); true = np.asarray(true)
    return {'RMSE': float(np.sqrt(mean_squared_error(true, pred))),
            'MAE': float(mean_absolute_error(true, pred)),
            'R2': float(r2_score(true, pred)),
            'Bias': float(np.mean(pred - true)), 'N': len(true)}


def encode_global(zwd, lat, lon, doy, hour):
    return np.array([zwd,
                     np.sin(np.deg2rad(lat)), np.cos(np.deg2rad(lat)),
                     np.sin(np.deg2rad(lon)), np.cos(np.deg2rad(lon)),
                     np.sin(2*np.pi*doy/365.0), np.cos(2*np.pi*doy/365.0),
                     np.sin(2*np.pi*hour/24.0), np.cos(2*np.pi*hour/24.0)], dtype=np.float32)


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'设备: {device}', flush=True)

    # 1) 第一阶段 Tm GRU 预测 (2017)
    tm_rows = []
    for fp in sorted(glob.glob(TM_PRED)):
        d = pd.read_csv(fp, sep='\s+')
        tm_rows.append(d)
    tmdf = pd.concat(tm_rows, ignore_index=True)
    tmdf.columns = [c.strip() for c in tmdf.columns]
    print(f'Tm GRU 预测: {len(tmdf)} 条, 站数={tmdf["StationID"].nunique()}', flush=True)

    # 2) 第二阶段模型 + scalers
    from model import ProfileTransformer
    ckpt = torch.load(os.path.join(MODEL_DIR, 'best_model.pth'), map_location=device, weights_only=True)
    model = ProfileTransformer(
        d_model=ckpt.get('args', {}).get('d_model', 128),
        n_heads=ckpt.get('args', {}).get('n_heads', 8),
        n_layers=ckpt.get('args', {}).get('n_layers', 4),
        ff_dim=ckpt.get('args', {}).get('ff_dim', 512),
        dropout=ckpt.get('args', {}).get('dropout', 0.1),
        global_feat_dim=9).to(device)
    model.load_state_dict(ckpt['model_state_dict']); model.eval()
    with open(os.path.join(MODEL_DIR, 'scalers.pkl'), 'rb') as f:
        scalers = pickle.load(f)
    ls = scalers['level_scaler']; gs = scalers['global_scaler']
    h_mean = scalers['height_mean']; h_std = scalers['height_std']

    # 3) GPT3
    from gpt3 import GPT3
    gpt3 = GPT3(GRD)

    # 4) 逐站读取 2017 数据: 按"时间戳"构建廓线(00/12各一条), Tm GRU 按 DOY 内顺序匹配
    records = []
    for sid, g in tmdf.groupby('StationID'):
        fp = f'{DATA}/2017/{sid}_met.txt'
        if not os.path.exists(fp):
            continue
        df = pd.read_csv(fp, header=0, sep=None, engine='python')
        fc = df.columns[0]
        if str(fc).startswith('Unnamed') or str(fc).strip() == '':
            df = df.rename(columns={fc: 'TIME'})
        df['TIME'] = pd.to_datetime(df['TIME'], errors='coerce')
        for c in ['ELV', 'TS', 'PS', 'WPS', 'ZWD', 'PWV', 'Tm', 'LAT', 'LON']:
            df[c] = pd.to_numeric(df[c], errors='coerce')
        df = df.dropna(subset=['ELV', 'TS', 'PS', 'WPS', 'ZWD', 'PWV', 'TIME'])
        df['doy'] = df['TIME'].dt.dayofyear
        lat = float(df['LAT'].iloc[0]); lon = float(df['LON'].iloc[0])

        # Tm GRU 预测按 DOY 分组(每 DOY 可能有 1-2 条: 00/12 两次观测)
        tm_by_doy = {}
        for _, row in g.iterrows():
            tm_by_doy.setdefault(int(row['DOY']), []).append(float(row['Predict']))
        # 每个 DOY 的时间戳顺序
        doy_times = {}
        for ts in sorted(df['TIME'].unique()):
            doy_times.setdefault(ts.dayofyear, []).append(ts)

        for ts, sgroup in df.groupby(df['TIME']):
            d = ts.dayofyear
            tms = tm_by_doy.get(d)
            if not tms:
                continue
            # 该时间戳在当日观测中的序号 -> 匹配对应 Tm 预测
            times_today = doy_times.get(d, [])
            k = times_today.index(ts) if ts in times_today else 0
            tm_gru = tms[k] if k < len(tms) else float(np.mean(tms))
            sub = sgroup.sort_values('ELV')
            surf = sub.iloc[0]
            zwd = float(surf['ZWD']); pwv_t = float(surf['PWV']); elv = float(surf['ELV'])
            hour = float(ts.hour)
            # Transformer 输入: 该时刻整条廓线
            prof = sub[['ELV', 'TS', 'PS', 'WPS']].values.astype(np.float32)
            if len(prof) < 2:
                continue
            levels_n = ls.transform(prof).astype(np.float32)
            heights_n = ((prof[:, 0] - h_mean) / h_std).astype(np.float32)
            gf = gs.transform(encode_global(zwd, lat, lon, d, hour).reshape(1, -1))[0].astype(np.float32)
            L = len(prof)
            with torch.no_grad():
                pi = model(torch.from_numpy(levels_n).unsqueeze(0).to(device),
                           torch.from_numpy(heights_n).unsqueeze(0).to(device),
                           torch.from_numpy(gf).unsqueeze(0).to(device),
                           torch.ones(1, L, dtype=torch.bool, device=device)).item()
            pwv_trans = pi * zwd
            # GPT3 Tm
            _, _, tm_gpt3, _ = gpt3.compute(lat, lon, elv, 2017, d, hour)
            tm_gpt3_v = float(np.asarray(tm_gpt3).item())
            pwv_gpt3 = pi_from_tm(tm_gpt3_v) * zwd
            pwv_gru_tm = pi_from_tm(tm_gru) * zwd
            tm_true = float(surf['Tm']) if 'Tm' in surf and not pd.isna(surf['Tm']) else np.nan
            records.append({'station': sid, 'doy': d, 'zwd': zwd, 'pwv_true': pwv_t,
                            'tm_true': tm_true, 'tm_gru': tm_gru, 'tm_gpt3': tm_gpt3_v,
                            'pwv_gru_tm': pwv_gru_tm, 'pwv_gpt3': pwv_gpt3, 'pwv_trans': pwv_trans})
        print(f'  站 {sid} 完成, 累计 {len(records)}', flush=True)

    df = pd.DataFrame(records)
    print(f'\n端到端匹配样本: {len(df)}', flush=True)
    df.to_csv('result_endtoend.csv', index=False, float_format='%.6f')

    print('\n===== 端到端两步法对比 (2017, 同一批样本) =====')
    comps = [('传统两步法(GRU-Tm)', 'pwv_gru_tm'), ('GPT3', 'pwv_gpt3'), ('Transformer', 'pwv_trans')]
    rows = []
    for name, col in comps:
        m = metrics(df[col].values, df['pwv_true'].values)
        rows.append({'method': name, **m})
        print(f'{name:22s}: RMSE={m["RMSE"]:.4f}  MAE={m["MAE"]:.4f}  R2={m["R2"]:.6f}  Bias={m["Bias"]:.4f}')
    out = pd.DataFrame(rows)
    out.to_csv('result_endtoend_metrics.csv', index=False, float_format='%.6f')

    # Tm 误差对比 (相对数据真值 Tm)
    sub = df.dropna(subset=['tm_true'])
    if len(sub) > 0:
        e_gru = sub['tm_gru'].values - sub['tm_true'].values
        e_gpt3 = sub['tm_gpt3'].values - sub['tm_true'].values
        print(f'\nTm 误差 (真值来自探空, N={len(sub)}):')
        print(f'  第一阶段 GRU: RMSE={np.sqrt((e_gru**2).mean()):.3f} K  Bias={e_gru.mean():+.3f} K')
        print(f'  GPT3       : RMSE={np.sqrt((e_gpt3**2).mean()):.3f} K  Bias={e_gpt3.mean():+.3f} K')

    # 散点图
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    for ax, (name, col) in zip(axes, comps):
        m = metrics(df[col].values, df['pwv_true'].values)
        ax.scatter(df['pwv_true'], df[col], s=3, alpha=0.2, color='steelblue')
        v0, v1 = df['pwv_true'].min(), df['pwv_true'].max()
        ax.plot([v0, v1], [v0, v1], 'r--', lw=1)
        ax.set_title(f'{name}  RMSE={m["RMSE"]:.3f} R2={m["R2"]:.4f}')
        ax.set_xlabel('True PWV (mm)'); ax.set_ylabel('Pred PWV (mm)'); ax.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig('result_endtoend_scatter.png', dpi=200); plt.close()
    print('\n完成 -> result_endtoend*.csv / result_endtoend_scatter.png')


if __name__ == '__main__':
    main()
