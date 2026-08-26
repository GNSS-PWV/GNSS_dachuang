# -*- coding: utf-8 -*-
"""
层贡献诊断 (廓线可解释性): 遮挡各高度层 -> dPi, 找出对转换系数 Pi 最重要的高度层
输入: 已训练模型 + 武汉探空廓线(真实) + scalers
"""
import os, sys, pickle
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
OUT = os.path.dirname(os.path.abspath(__file__))
BASE = r'D:\gnss水汽反演'

MODEL_PTH = os.path.join(OUT, 'result_aligned_official', 'best_model.pth')
SCALERS = os.path.join(OUT, 'result_aligned_official', 'scalers.pkl')
RAD = os.path.join(OUT, 'radiosonde_2019', 'CHM00057494_met.txt')


def encode_global(zwd, lat, lon, doy, hour):
    return np.array([zwd,
                     np.sin(np.deg2rad(lat)), np.cos(np.deg2rad(lat)),
                     np.sin(np.deg2rad(lon)), np.cos(np.deg2rad(lon)),
                     np.sin(2*np.pi*doy/365.0), np.cos(2*np.pi*doy/365.0),
                     np.sin(2*np.pi*hour/24.0), np.cos(2*np.pi*hour/24.0)], dtype=np.float32)


def main():
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

    # 读取武汉探空, 挑代表性廓线: 夏季湿润(6月) 与 冬季干燥(1月)
    df = pd.read_csv(RAD, header=0, sep=None, engine='python')
    fc = df.columns[0]
    if str(fc).startswith('Unnamed') or str(fc).strip() == '':
        df = df.rename(columns={fc: 'TIME'})
    df['TIME'] = pd.to_datetime(df['TIME'], errors='coerce')
    for c in ['ELV', 'TS', 'PS', 'WPS', 'ZWD', 'PWV', 'LAT', 'LON']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df = df.dropna(subset=['ELV', 'TS', 'PS', 'WPS'])
    lat = float(df['LAT'].iloc[0]); lon = float(df['LON'].iloc[0])
    # 挑两个时刻: 1月15日 00Z, 6月20日 12Z (暴雨前后)
    picks = [('2019-01-15 00:00:00', 15, 0), ('2019-06-20 12:00:00', 171, 12)]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, (ts, doy, hour) in zip(axes, picks):
        sub = df[df['TIME'] == ts].sort_values('ELV')
        if len(sub) < 5:
            print('无该时刻', ts); continue
        prof = sub[['ELV', 'TS', 'PS', 'WPS']].values.astype(np.float32)
        zwd = float(sub['ZWD'].iloc[0])
        levels_n = ls.transform(prof).astype(np.float32)
        heights_n = ((prof[:, 0] - h_mean) / h_std).astype(np.float32)
        gf = gs.transform(encode_global(zwd, lat, lon, doy, hour).reshape(1, -1))[0].astype(np.float32)
        L = len(prof)
        with torch.no_grad():
            pi0 = model(torch.from_numpy(levels_n).unsqueeze(0).to(device),
                        torch.from_numpy(heights_n).unsqueeze(0).to(device),
                        torch.from_numpy(gf).unsqueeze(0).to(device),
                        torch.ones(1, L, dtype=torch.bool, device=device)).item()
        # 遮挡每个高度带 (0-2,2-4,4-6,6-8,8-10,10-15,15km+)
        bands = [(0, 2), (2, 4), (4, 6), (6, 8), (8, 10), (10, 15), (15, 99)]
        labels = ['0-2', '2-4', '4-6', '6-8', '8-10', '10-15', '15+']
        dpi = []
        for (lo, hi) in bands:
            mask = ~((prof[:, 0] >= lo*1000) & (prof[:, 0] < hi*1000))
            lv2 = levels_n[mask]
            h2 = heights_n[mask]
            with torch.no_grad():
                pi1 = model(torch.from_numpy(lv2).unsqueeze(0).to(device),
                            torch.from_numpy(h2).unsqueeze(0).to(device),
                            torch.from_numpy(gf).unsqueeze(0).to(device),
                            torch.ones(1, len(lv2), dtype=torch.bool, device=device)).item()
            dpi.append(pi0 - pi1)
        ax.bar(labels, dpi, color='steelblue')
        ax.axhline(0, color='k', lw=0.8)
        ax.set_xlabel('Occluded height band (km)')
        ax.set_ylabel('dPi (baseline - occluded)')
        ax.set_title(f'{ts}\nPi_baseline={pi0:.4f}  PWV={sub["PWV"].iloc[0]:.1f}mm')
        ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, 'layer_contribution.png'), dpi=150)
    print('已保存 layer_contribution.png')


if __name__ == '__main__':
    main()
