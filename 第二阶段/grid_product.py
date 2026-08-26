# -*- coding: utf-8 -*-
"""
全球格网化转换系数 Pi 产品 (创新点三)

用训练好的 ProfileTransformer 在经纬度网格上推理, 生成"大气垂直结构 -> 转换系数 Pi"的
全球格网产品. 每个格网点用最近 K 个探空站的分季节平均廓线作为代表性廓线输入模型.

输出:
  result_grid/pi_grid_<season>.csv   (lat, lon, pi, zwd, n_stations)
  result_grid/pi_grid_annual.csv
  result_grid/pi_map_*.png           全球 Pi 分布图

用法:
  python grid_product.py --model_dir result --data_dir <xg_data> --out result_grid
"""
import os
import sys
import glob
import pickle
import argparse
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

LEVEL_FEATURES = ['ELV', 'TS', 'PS', 'WPS']
SEASONS = {'DJF': 15, 'MAM': 105, 'JJA': 198, 'SON': 288}
MONTHS = {'M%02d' % m: doy for m, doy in [
    (1, 15), (2, 45), (3, 74), (4, 105), (5, 135), (6, 166),
    (7, 196), (8, 227), (9, 258), (10, 288), (11, 319), (12, 349)]}
BIN_WIDTH = 1000.0      # 高度分箱宽度 (m)
MAX_HEIGHT = 30000.0    # 最大高度 (m)


def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    p1 = np.deg2rad(lat1); p2 = np.deg2rad(lat2)
    dp = np.deg2rad(lat2 - lat1); dl = np.deg2rad(lon2 - lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def encode_global(zwd, lat, lon, doy, hour):
    return np.array([
        zwd,
        np.sin(np.deg2rad(lat)), np.cos(np.deg2rad(lat)),
        np.sin(np.deg2rad(lon)), np.cos(np.deg2rad(lon)),
        np.sin(2 * np.pi * doy / 365.0), np.cos(2 * np.pi * doy / 365.0),
        np.sin(2 * np.pi * hour / 24.0), np.cos(2 * np.pi * hour / 24.0),
    ], dtype=np.float32)


def build_station_seasonal_profiles(data_dir, cache=None, monthly=False):
    """读取所有站点文件, 计算每站每季(或每月)的按高度分箱平均廓线.

    参数:
      monthly: True 时按月份(M01-M12)分组, 否则按季节(DJF/MAM/JJA/SON)

    返回: dict {station_id: {'lat':.., 'lon':.., 'elv_surface':..,
                             'season'/'month': {period: (levels, zwd_mean), ...}}}
          levels: (n_bins, 4) = [ELV(箱中心), TS, PS, WPS]
    """
    if cache and os.path.exists(cache):
        print(f'加载站点平均廓线缓存: {cache}', flush=True)
        with open(cache, 'rb') as f:
            return pickle.load(f)

    files = sorted(glob.glob(os.path.join(data_dir, '**', '*_met.txt'), recursive=True))
    if not files:
        files = sorted(glob.glob(os.path.join(data_dir, '*_met.txt')))
    print(f'读取 {len(files)} 个站点文件...', flush=True)

    from data import COLUMN_NAMES
    st = {}
    for i, fp in enumerate(files):
        sid = os.path.basename(fp).split('_met')[0]
        try:
            df = pd.read_csv(fp, header=0, sep=None, engine='python')
            fc = df.columns[0]
            if str(fc).startswith('Unnamed') or str(fc).strip() == '':
                df = df.rename(columns={fc: 'TIME'})
            df['TIME'] = pd.to_datetime(df['TIME'], errors='coerce')
            for c in ['ELV', 'TS', 'PS', 'WPS', 'ZWD', 'LAT', 'LON']:
                df[c] = pd.to_numeric(df[c], errors='coerce')
            df = df.dropna(subset=['ELV', 'TS', 'PS', 'WPS', 'ZWD', 'LAT', 'LON', 'TIME'])
        except Exception:
            continue
        if len(df) == 0:
            continue
        lat = float(df['LAT'].iloc[0]); lon = float(df['LON'].iloc[0])
        elv_s = float(df.loc[df['TIME'].idxmin(), 'ELV']) if False else float(df.sort_values('ELV').iloc[0]['ELV'])
        if monthly:
            df['period'] = 'M' + df['TIME'].dt.month.astype(str).str.zfill(2)
        else:
            df['period'] = df['TIME'].dt.month.map({12: 'DJF', 1: 'DJF', 2: 'DJF',
                                                    3: 'MAM', 4: 'MAM', 5: 'MAM',
                                                    6: 'JJA', 7: 'JJA', 8: 'JJA',
                                                    9: 'SON', 10: 'SON', 11: 'SON'})
        pkey = 'month' if monthly else 'season'
        st[sid] = {'lat': lat, 'lon': lon, 'elv_surface': elv_s, pkey: {}}
        for season, g in df.groupby('period'):
            bins = np.arange(0, MAX_HEIGHT + BIN_WIDTH, BIN_WIDTH)
            idx = np.clip((g['ELV'].values // BIN_WIDTH).astype(int), 0, len(bins) - 2)
            centers = bins[:-1] + BIN_WIDTH / 2
            ts = np.full(len(bins) - 1, np.nan)
            ps = np.full(len(bins) - 1, np.nan)
            wps = np.full(len(bins) - 1, np.nan)
            for b in range(len(bins) - 1):
                m = idx == b
                if m.sum() > 0:
                    ts[b] = g['TS'].values[m].mean()
                    ps[b] = g['PS'].values[m].mean()
                    wps[b] = g['WPS'].values[m].mean()
            ok = ~np.isnan(ts)
            levels = np.stack([centers[ok], ts[ok], ps[ok], wps[ok]], axis=1).astype(np.float32)
            zwd_mean = float(g['ZWD'].mean())
            st[sid][pkey][season] = (levels, zwd_mean)
        if (i + 1) % 200 == 0:
            print(f'  已处理 {i+1}/{len(files)} 站', flush=True)

    if cache:
        with open(cache, 'wb') as f:
            pickle.dump(st, f)
        print(f'站点季节平均廓线已缓存: {cache}', flush=True)
    return st


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model_dir', default='result', help='含 best_model.pth + scalers.pkl 的目录')
    ap.add_argument('--data_dir', required=True)
    ap.add_argument('--out', default='result_grid')
    ap.add_argument('--grid_step', type=float, default=5.0, help='格网步长(度)')
    ap.add_argument('--k', type=int, default=5, help='每个格网点使用的最近站数')
    ap.add_argument('--cache', default=None, help='站点平均廓线缓存文件')
    ap.add_argument('--max_files', type=int, default=None, help='(调试)只读前 N 个文件')
    ap.add_argument('--monthly', action='store_true', help='按月(M01-M12)而非按季节生成格网')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'设备: {device}', flush=True)

    # 加载模型与 scalers
    from model import ProfileTransformer
    ckpt = torch.load(os.path.join(args.model_dir, 'best_model.pth'), map_location=device, weights_only=True)
    model = ProfileTransformer(
        d_model=ckpt.get('args', {}).get('d_model', 128),
        n_heads=ckpt.get('args', {}).get('n_heads', 8),
        n_layers=ckpt.get('args', {}).get('n_layers', 4),
        ff_dim=ckpt.get('args', {}).get('ff_dim', 512),
        dropout=ckpt.get('args', {}).get('dropout', 0.1),
        global_feat_dim=9,
    ).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    with open(os.path.join(args.model_dir, 'scalers.pkl'), 'rb') as f:
        scalers = pickle.load(f)
    level_scaler = scalers['level_scaler']
    global_scaler = scalers['global_scaler']
    h_mean = scalers['height_mean']; h_std = scalers['height_std']
    print(f'模型加载完成: {os.path.join(args.model_dir, "best_model.pth")}', flush=True)

    # 站点平均廓线
    st = build_station_seasonal_profiles(args.data_dir, cache=args.cache, monthly=args.monthly)
    period_key = 'month' if args.monthly else 'season'
    periods = MONTHS if args.monthly else SEASONS
    if args.max_files:
        keys = list(st.keys())[:args.max_files]
        st = {k: st[k] for k in keys}
    print(f'参与站点: {len(st)}', flush=True)
    station_ids = list(st.keys())
    st_lat = np.array([st[s]['lat'] for s in station_ids])
    st_lon = np.array([st[s]['lon'] for s in station_ids])

    # 全球格网
    lats = np.arange(-85, 86, args.grid_step)
    lons = np.arange(-180, 180, args.grid_step)
    print(f'格网: {len(lats)} x {len(lons)} = {len(lats)*len(lons)} 点', flush=True)

    all_rows = []
    for season, doy in periods.items():
        pi_map = np.full((len(lats), len(lons)), np.nan)
        zwd_map = np.full((len(lats), len(lons)), np.nan)
        nst_map = np.zeros((len(lats), len(lons)), dtype=int)
        with torch.no_grad():
            for i, lat in enumerate(lats):
                for j, lon in enumerate(lons):
                    # 最近 K 站
                    d = haversine(st_lat, st_lon, lat, lon)
                    idx = np.argsort(d)[:args.k]
                    ks = [station_ids[k] for k in idx]
                    levels_list, zwd_list = [], []
                    for s in ks:
                        if season in st[s][period_key]:
                            lv, zw = st[s][period_key][season]
                            if len(lv) > 0:
                                levels_list.append(lv)
                                zwd_list.append(zw)
                    if len(levels_list) == 0:
                        continue
                    # 按高度箱对齐平均
                    all_elv = np.unique(np.concatenate([lv[:, 0] for lv in levels_list]))
                    ts = np.full(len(all_elv), np.nan); ps = np.full(len(all_elv), np.nan); wps = np.full(len(all_elv), np.nan)
                    for lv in levels_list:
                        pos = np.searchsorted(all_elv, lv[:, 0])
                        ts[pos] = np.nanmean([ts[pos], lv[:, 1]], axis=0)
                        ps[pos] = np.nanmean([ps[pos], lv[:, 2]], axis=0)
                        wps[pos] = np.nanmean([wps[pos], lv[:, 3]], axis=0)
                    ok = ~np.isnan(ts)
                    levels = np.stack([all_elv[ok], ts[ok], ps[ok], wps[ok]], axis=1).astype(np.float32)
                    zwd_mean = float(np.mean(zwd_list))
                    n_levels = len(levels)
                    # 归一化
                    levels_n = level_scaler.transform(levels).astype(np.float32)
                    heights_n = ((levels[:, 0] - h_mean) / h_std).astype(np.float32)
                    gf = global_scaler.transform(encode_global(zwd_mean, lat, lon, doy, 12).reshape(1, -1))[0].astype(np.float32)
                    # 组织 batch (L=1)
                    B = 1
                    L = n_levels
                    levels_t = torch.from_numpy(levels_n).unsqueeze(0).to(device)
                    heights_t = torch.from_numpy(heights_n).unsqueeze(0).to(device)
                    gf_t = torch.from_numpy(gf).unsqueeze(0).to(device)
                    mask_t = torch.ones(B, L, dtype=torch.bool, device=device)
                    pi = model(levels_t, heights_t, gf_t, mask_t).item()
                    pi_map[i, j] = pi
                    zwd_map[i, j] = zwd_mean
                    nst_map[i, j] = len(levels_list)
        # 保存
        for i, lat in enumerate(lats):
            for j, lon in enumerate(lons):
                if not np.isnan(pi_map[i, j]):
                    all_rows.append({'lat': lat, 'lon': lon, 'season': season,
                                     'pi': pi_map[i, j], 'zwd': zwd_map[i, j],
                                     'n_stations': int(nst_map[i, j])})
        # 地图
        lon_m, lat_m = np.meshgrid(lons, lats)
        ok = ~np.isnan(pi_map)
        fig, ax = plt.subplots(figsize=(12, 6))
        sc = ax.scatter(lon_m[ok], lat_m[ok], c=pi_map[ok], cmap='viridis', s=20, vmin=0.12, vmax=0.20)
        ax.set_title(f'Global gridded Pi (ProfileTransformer) - {season}')
        ax.set_xlabel('Lon'); ax.set_ylabel('Lat')
        plt.colorbar(sc, label='Pi (PWV/ZWD)')
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(args.out, f'pi_map_{season}.png'), dpi=200)
        plt.close()
        print(f'[season {season}] 完成, 有效格点: {np.sum(ok)}', flush=True)

    df = pd.DataFrame(all_rows)
    df.to_csv(os.path.join(args.out, 'pi_grid_all.csv'), index=False, float_format='%.6f')
    # 年平均
    annual = df.groupby(['lat', 'lon']).agg(pi=('pi', 'mean'), zwd=('zwd', 'mean'),
                                            n_stations=('n_stations', 'mean')).reset_index()
    annual.to_csv(os.path.join(args.out, 'pi_grid_annual.csv'), index=False, float_format='%.6f')
    print(f'\n格网化 Pi 产品已保存: {args.out}')
    print(f'Pi 范围: [{df["pi"].min():.4f}, {df["pi"].max():.4f}]')


if __name__ == '__main__':
    main()
