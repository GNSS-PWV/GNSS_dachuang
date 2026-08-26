# -*- coding: utf-8 -*-
"""
端到端部署模拟: 无探空廓线时用"气候态平均廓线"能否部署?

场景: GNSS 站点没有本地探空, 部署时用最近训练站的季节平均廓线作为模型输入.
对每个(测试站, 季节), 用最近 K 个训练站的季节平均廓线构造气候态廓线 -> 模型 -> Pi_clim,
再应用到该站该季所有观测 (PWV_clim = Pi_clim * ZWD_true).

对比:
  pwv_pred   : 用真实探空廓线 (当前 RMSE)
  pwv_clim   : 用气候态廓线 (部署场景)
  pwv_gpt3   : GPT3 经验 Tm 基线

用法:
  python deploy_simulation.py --csv test_predictions.csv --model_dir result_aligned \
      --data_dir <xg_data> --cache st_seasonal_cache.pkl --test_stations test_stations_official_36.txt
"""
import os
import sys
import glob
import pickle
import argparse
import numpy as np
import pandas as pd
import datetime as dt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

SEASONS = {'DJF': 15, 'MAM': 105, 'JJA': 198, 'SON': 288}
BIN_WIDTH = 1000.0
MAX_HEIGHT = 30000.0


def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    p1 = np.deg2rad(lat1); p2 = np.deg2rad(lat2)
    dp = np.deg2rad(lat2 - lat1); dl = np.deg2rad(lon2 - lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def encode_global(zwd, lat, lon, doy, hour):
    return np.array([zwd,
                     np.sin(np.deg2rad(lat)), np.cos(np.deg2rad(lat)),
                     np.sin(np.deg2rad(lon)), np.cos(np.deg2rad(lon)),
                     np.sin(2 * np.pi * doy / 365.0), np.cos(2 * np.pi * doy / 365.0),
                     np.sin(2 * np.pi * hour / 24.0), np.cos(2 * np.pi * hour / 24.0)], dtype=np.float32)


def metrics(pred, true):
    pred = np.asarray(pred); true = np.asarray(true)
    rmse = float(np.sqrt(mean_squared_error(true, pred)))
    mae = float(mean_absolute_error(true, pred))
    r2 = float(r2_score(true, pred))
    bias = float(np.mean(pred - true))
    return {'RMSE': rmse, 'MAE': mae, 'R2': r2, 'Bias': bias, 'N': len(true)}


def season_of(month):
    if month in (12, 1, 2): return 'DJF'
    if month in (3, 4, 5): return 'MAM'
    if month in (6, 7, 8): return 'JJA'
    return 'SON'


def mean_profile(levels_list):
    """多个站季节平均廓线按高度箱对齐求平均."""
    if not levels_list:
        return None
    all_elv = np.unique(np.concatenate([lv[:, 0] for lv in levels_list]))
    ts = np.full(len(all_elv), np.nan); ps = np.full(len(all_elv), np.nan); wps = np.full(len(all_elv), np.nan)
    for lv in levels_list:
        pos = np.searchsorted(all_elv, lv[:, 0])
        ts[pos] = np.nanmean([ts[pos], lv[:, 1]], axis=0)
        ps[pos] = np.nanmean([ps[pos], lv[:, 2]], axis=0)
        wps[pos] = np.nanmean([wps[pos], lv[:, 3]], axis=0)
    ok = ~np.isnan(ts)
    return np.stack([all_elv[ok], ts[ok], ps[ok], wps[ok]], axis=1).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', required=True)
    ap.add_argument('--model_dir', default='result_aligned')
    ap.add_argument('--data_dir', required=True)
    ap.add_argument('--cache', required=True, help='站点季节平均廓线缓存(由 grid_product 生成)')
    ap.add_argument('--test_stations', required=True, help='官方测试站名单, 用于排除训练站')
    ap.add_argument('--out', default='result_deploy')
    ap.add_argument('--k', type=int, default=5)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    from model import ProfileTransformer
    ckpt = torch.load(os.path.join(args.model_dir, 'best_model.pth'), map_location=device, weights_only=True)
    model = ProfileTransformer(
        d_model=ckpt.get('args', {}).get('d_model', 128),
        n_heads=ckpt.get('args', {}).get('n_heads', 8),
        n_layers=ckpt.get('args', {}).get('n_layers', 4),
        ff_dim=ckpt.get('args', {}).get('ff_dim', 512),
        dropout=ckpt.get('args', {}).get('dropout', 0.1),
        global_feat_dim=9).to(device)
    model.load_state_dict(ckpt['model_state_dict']); model.eval()
    with open(os.path.join(args.model_dir, 'scalers.pkl'), 'rb') as f:
        scalers = pickle.load(f)
    ls = scalers['level_scaler']; gs = scalers['global_scaler']
    h_mean = scalers['height_mean']; h_std = scalers['height_std']
    print(f'模型加载: {args.model_dir}', flush=True)

    with open(args.cache, 'rb') as f:
        st = pickle.load(f)
    test_ids = set()
    with open(args.test_stations, 'r', encoding='utf-8') as f:
        for ln in f:
            ln = ln.strip()
            if ln and not ln.startswith('#'):
                test_ids.add(ln.split()[0])
    train_ids = [s for s in st.keys() if s not in test_ids]
    print(f'训练站数(用于气候态廓线): {len(train_ids)}', flush=True)
    tl = np.array([st[s]['lat'] for s in train_ids])
    tn = np.array([st[s]['lon'] for s in train_ids])

    df = pd.read_csv(args.csv)
    df['month'] = pd.to_datetime(df['time']).dt.month
    df['season'] = df['month'].apply(season_of)
    df['hour'] = pd.to_datetime(df['time']).dt.hour
    print(f'观测: {len(df)}, 测试站: {df["station_id"].nunique()}', flush=True)

    # 对每个(测试站, 季节): 气候态廓线 -> Pi_clim
    pi_clim_map = {}
    with torch.no_grad():
        for (sid, season), g in df.groupby(['station_id', 'season']):
            lat0 = g['lat'].iloc[0]; lon0 = g['lon'].iloc[0]
            d = haversine(tl, tn, lat0, lon0)
            idx = np.argsort(d)[:args.k]
            levels_list = []
            for k in idx:
                if season in st[train_ids[k]]['season']:
                    lv, _ = st[train_ids[k]]['season'][season]
                    if len(lv) > 0:
                        levels_list.append(lv)
            prof = mean_profile(levels_list)
            if prof is None:
                pi_clim_map[(sid, season)] = np.nan
                continue
            levels_n = ls.transform(prof).astype(np.float32)
            heights_n = ((prof[:, 0] - h_mean) / h_std).astype(np.float32)
            doy = SEASONS[season]
            zwd_ref = float(g['zwd'].mean())
            gf = gs.transform(encode_global(zwd_ref, lat0, lon0, doy, 12).reshape(1, -1))[0].astype(np.float32)
            L = len(prof)
            pi = model(torch.from_numpy(levels_n).unsqueeze(0).to(device),
                       torch.from_numpy(heights_n).unsqueeze(0).to(device),
                       torch.from_numpy(gf).unsqueeze(0).to(device),
                       torch.ones(1, L, dtype=torch.bool, device=device)).item()
            pi_clim_map[(sid, season)] = pi

    df['pi_clim'] = df.apply(lambda r: pi_clim_map.get((r['station_id'], r['season']), np.nan), axis=1)
    df['pwv_clim'] = df['pi_clim'] * df['zwd']
    df = df.dropna(subset=['pwv_clim'])
    print(f'有效样本: {len(df)}', flush=True)

    m_true = metrics(df['pwv_pred'].values, df['pwv_true'].values)
    m_clim = metrics(df['pwv_clim'].values, df['pwv_true'].values)

    print('\n=== 部署模拟: 真实廓线 vs 气候态廓线 ===')
    for name, m in [('真实探空廓线 (pwv_pred)', m_true), ('气候态廓线 (pwv_clim)', m_clim)]:
        print(f'{name:28s}: RMSE={m["RMSE"]:.4f}  MAE={m["MAE"]:.4f}  Bias={m["Bias"]:.4f}  R2={m["R2"]:.6f}')
    print(f'\n部署退化 (clim vs radiosonde): RMSE {m_true["RMSE"]:.4f} -> {m_clim["RMSE"]:.4f} mm')

    df.to_csv(os.path.join(args.out, 'deploy_predictions.csv'), index=False, float_format='%.6f')
    pd.DataFrame([{'profile': 'radiosonde', **m_true}, {'profile': 'climatology', **m_clim}]).to_csv(
        os.path.join(args.out, 'deploy_metrics.csv'), index=False, float_format='%.6f')

    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.scatter(df['pwv_true'], df['pwv_clim'], s=4, alpha=0.2, color='darkorange')
    v0 = df['pwv_true'].min(); v1 = df['pwv_true'].max()
    ax.plot([v0, v1], [v0, v1], 'r--', lw=1)
    ax.set_xlabel('True PWV (mm)'); ax.set_ylabel('Climatology-profile PWV (mm)')
    ax.set_title(f'Deploy simulation (climatological profile)\nRMSE={m_clim["RMSE"]:.3f} mm')
    ax.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(args.out, 'deploy_scatter.png'), dpi=200); plt.close()
    print(f'\n完成 -> {args.out}')


if __name__ == '__main__':
    main()
