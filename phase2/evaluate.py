# -*- coding: utf-8 -*-
"""
第二阶段评估脚本: Transformer vs Saastamoinen 基线对比

对比内容:
  1. Saastamoinen (full): 从 P/T/e 计算 ZTD->ZHD->ZWD, 再用 Tm 算 Pi -> PWV
  2. Saastamoinen (Pi-only): 用数据中真值 ZWD, 仅用 Tm 算 Pi -> PWV (隔离转换系数误差)
  3. ProfileTransformer: 用数据中真值 ZWD, 用廓线算 Pi -> PWV (本阶段模型)

输出:
  - 对比指标表 (RMSE/MAE/R2/Bias)
  - 并排散点图
  - 误差分布直方图
  - 逐站点指标

用法:
  python evaluate.py --data_dir <data> --model_path result/best_model.pth --output_dir result
"""
import os
import sys
import glob
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 环境变量
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# 导入路径
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)
parent_dir = os.path.dirname(script_dir)
sys.path.insert(0, parent_dir)

from data import read_station_file, ProfileDataset, collate_profiles, station_based_split, _encode_global, GLOBAL_FEATURE_DIM
from model import ProfileTransformer
from saastamoinen_pwv import saastamoinen_pwv_batch

import torch
import warnings
warnings.filterwarnings('ignore')

# 物理常数
K2P = 22.1
K3 = 3.739e5
RV = 461.495
RHO_W = 1000.0


def compute_metrics(pred, true):
    pred = np.asarray(pred, dtype=np.float64).flatten()
    true = np.asarray(true, dtype=np.float64).flatten()
    rmse = float(np.sqrt(mean_squared_error(true, pred)))
    mae = float(mean_absolute_error(true, pred))
    r2 = float(r2_score(true, pred))
    bias = float(np.mean(pred - true))
    rel = rmse / np.mean(true) * 100 if np.mean(true) > 0 else 0.0
    return {'RMSE': rmse, 'MAE': mae, 'R2': r2, 'Bias': bias, 'Rel_RMSE_pct': rel, 'N': len(true)}


def saastamoinen_pi(Tm_K):
    """从 Tm 计算转换系数 Pi (无量纲)."""
    return 1e8 / (RHO_W * RV * (K3 / Tm_K + K2P))


def read_surface_from_profiles(profiles):
    """从廓线列表中提取地表层 Saastamoinen 所需的输入 (P/T/e/Tm/lat/h/ZWD/PWV)."""
    records = []
    for p in profiles:
        levels = p['levels']  # (n_levels, 4) = [ELV, TS, PS, WPS]
        surface = levels[0]   # 地表层 (最低 ELV)
        elv, ts, ps, wps = surface[0], surface[1], surface[2], surface[3]
        records.append({
            'station_id': p['station_id'],
            'time_str': p['time_str'],
            'lat': p['global_raw']['lat'],
            'lon': p['global_raw']['lon'],
            'elv': elv,
            'ps': ps,
            'ts': ts,
            'wps': wps,
            'zwd': p['zwd_surface'],
            'pwv': p['pwv_surface'],
        })
    return pd.DataFrame(records)


def evaluate(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'设备: {device}', flush=True)

    # 加载数据
    from data import load_all_profiles
    profiles = load_all_profiles(args.data_dir, max_files=args.max_files)
    if len(profiles) == 0:
        print('无数据!')
        return

    # 与训练时相同的切分 (确保测试集一致)
    _, _, test_profiles = station_based_split(
        profiles, args.test_station_ratio, args.val_ratio, args.seed,
        test_stations=args.test_stations
    )
    print(f'测试集: {len(test_profiles)} 条廓线', flush=True)

    if len(test_profiles) == 0:
        print('测试集为空! 请检查 test_station_ratio 或增加数据量.')
        return

    # --- Saastamoinen 基线 ---
    surface_df = read_surface_from_profiles(test_profiles)

    # 需要从原始数据中获取 Tm (Saastamoinen 需要 Tm, 但我们的廓线数据中没有保存 Tm)
    # 重新读取原始文件提取 Tm
    print('提取 Tm 用于 Saastamoinen 基线...', flush=True)
    # 从原始数据文件中获取 Tm: 读取所有测试站文件, 匹配时间戳
    test_stations = set(p['station_id'] for p in test_profiles)
    tm_lookup = {}
    for fp in sorted(glob.glob(os.path.join(args.data_dir, '**', '*_met.txt'), recursive=True)) + \
               sorted(glob.glob(os.path.join(args.data_dir, '*_met.txt'))):
        sid = os.path.basename(fp).split('_met')[0]
        if sid not in test_stations:
            continue
        try:
            df = pd.read_csv(fp, header=0, sep=None, engine='python')
            fc = df.columns[0]
            if str(fc).startswith('Unnamed') or str(fc).strip() == '':
                df = df.rename(columns={fc: 'TIME'})
            df['TIME'] = pd.to_datetime(df['TIME'], errors='coerce')
            df = df.dropna(subset=['TIME', 'Tm'])
            for ts, g in df.groupby('TIME', sort=True):
                g = g.sort_values('ELV')
                tm_lookup[(sid, str(ts))] = float(g['Tm'].iloc[0])
        except Exception:
            continue

    # 匹配 Tm
    tm_vals = []
    for _, row in surface_df.iterrows():
        key = (row['station_id'], row['time_str'])
        tm_vals.append(tm_lookup.get(key, np.nan))
    surface_df['tm'] = tm_vals
    surface_df = surface_df.dropna(subset=['tm'])
    print(f'  匹配到 Tm 的样本: {len(surface_df)}', flush=True)

    if len(surface_df) == 0:
        print('无法匹配 Tm, 跳过 Saastamoinen 基线对比')
        return

    lat = surface_df['lat'].values
    h = surface_df['elv'].values
    P = surface_df['ps'].values
    e = surface_df['wps'].values
    T = surface_df['ts'].values
    Tm = surface_df['tm'].values
    zwd_true = surface_df['zwd'].values
    pwv_true = surface_df['pwv'].values
    station_ids_saast = surface_df['station_id'].values

    # Saastamoinen full: 从 P/T/e 算 ZWD, 再算 PWV
    pwv_saast_full = saastamoinen_pwv_batch(lat, h, P, e, T, Tm)

    # Saastamoinen Pi-only: 用真值 ZWD, 仅算 Pi
    pi_saast = saastamoinen_pi(Tm)
    pwv_saast_pi = pi_saast * zwd_true

    m_saast_full = compute_metrics(pwv_saast_full, pwv_true)
    m_saast_pi = compute_metrics(pwv_saast_pi, pwv_true)

    print('\nSaastamoinen (full) PWV:', flush=True)
    print(f'  RMSE={m_saast_full["RMSE"]:.4f}  MAE={m_saast_full["MAE"]:.4f}  '
          f'R2={m_saast_full["R2"]:.4f}  Bias={m_saast_full["Bias"]:.4f}', flush=True)
    print('Saastamoinen (Pi-only) PWV:', flush=True)
    print(f'  RMSE={m_saast_pi["RMSE"]:.4f}  MAE={m_saast_pi["MAE"]:.4f}  '
          f'R2={m_saast_pi["R2"]:.4f}  Bias={m_saast_pi["Bias"]:.4f}', flush=True)

    # --- Transformer 模型 ---
    # 加载模型
    ckpt = torch.load(args.model_path, map_location=device, weights_only=True)
    model_args = ckpt.get('args', {})

    # scalers: 优先从训练时保存的 scalers.pkl 读取, 否则从训练数据重建
    scalers_pkl = os.path.join(os.path.dirname(args.model_path), 'scalers.pkl')
    if os.path.exists(scalers_pkl):
        import pickle
        with open(scalers_pkl, 'rb') as f:
            scalers = pickle.load(f)
        print(f'已加载 scalers: {scalers_pkl}', flush=True)
    else:
        print('未找到 scalers.pkl, 从训练数据重建...', flush=True)
        train_val_profiles = [p for p in profiles if p['station_id'] not in
                              set(sp['station_id'] for sp in test_profiles)]
        train_ds = ProfileDataset(train_val_profiles, fit_scalers=True)
        scalers = train_ds.get_scalers()
    test_ds = ProfileDataset(test_profiles, **scalers)
    model = ProfileTransformer(
        d_model=model_args.get('d_model', 128),
        n_heads=model_args.get('n_heads', 8),
        n_layers=model_args.get('n_layers', 4),
        ff_dim=model_args.get('ff_dim', 512),
        dropout=model_args.get('dropout', 0.1),
        global_feat_dim=model_args.get('global_feat_dim', GLOBAL_FEATURE_DIM),
    ).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print(f'已加载模型: {args.model_path}', flush=True)

    # 推理
    from torch.utils.data import DataLoader
    collate_fn = lambda batch: collate_profiles(batch, max_len=model_args.get('max_len', 30))
    test_loader = DataLoader(test_ds, batch_size=128, shuffle=False, collate_fn=collate_fn)

    all_pwv_pred, all_pwv_true_t, all_pi_pred, all_zwd_t, all_stations_t = [], [], [], [], []
    all_times_t, all_lats_t, all_lons_t, all_elvs_t, all_tms_t = [], [], [], [], []
    with torch.no_grad():
        for batch in test_loader:
            levels = batch['levels'].to(device)
            heights = batch['heights'].to(device)
            global_feat = batch['global_feat'].to(device)
            mask = batch['attention_mask'].to(device)
            zwd = batch['zwd'].to(device)

            pi_pred = model(levels, heights, global_feat, mask)
            pwv_pred = pi_pred * zwd

            all_pwv_pred.extend(pwv_pred.cpu().numpy())
            all_pwv_true_t.extend(batch['pwv'].numpy())
            all_pi_pred.extend(pi_pred.cpu().numpy())
            all_zwd_t.extend(zwd.cpu().numpy())
            all_stations_t.extend(batch['station_ids'])
            all_times_t.extend(batch['times'])
            all_lats_t.extend(batch['lats'].numpy())
            all_lons_t.extend(batch['lons'].numpy())
            all_elvs_t.extend(batch['elvs'].numpy())
            all_tms_t.extend(batch['tms'].numpy())

    pwv_pred = np.array(all_pwv_pred)
    pwv_true_t = np.array(all_pwv_true_t)
    pi_pred = np.array(all_pi_pred)
    zwd_t = np.array(all_zwd_t)
    stations_t = all_stations_t
    times_t = all_times_t
    lats_t = np.array(all_lats_t)
    lons_t = np.array(all_lons_t)
    elvs_t = np.array(all_elvs_t)
    tms_t = np.array(all_tms_t)

    # 保存测试集预测明细(含元数据, 供 GPT3 基线/第三阶段分析使用)
    import pandas as _pd
    pred_df = _pd.DataFrame({
        'station_id': stations_t, 'time': times_t,
        'lat': lats_t, 'lon': lons_t, 'elv': elvs_t, 'tm': tms_t,
        'zwd': zwd_t, 'pwv_true': pwv_true_t, 'pwv_pred': pwv_pred,
        'pwv_error': pwv_pred - pwv_true_t,
        'pi_true': pwv_true_t / zwd_t, 'pi_pred': pi_pred,
    })
    pred_csv = os.path.join(args.output_dir, 'test_predictions.csv')
    pred_df.to_csv(pred_csv, index=False, float_format='%.6f')
    print(f'测试集预测明细已保存: {pred_csv} (N={len(pred_df)})', flush=True)

    m_trans = compute_metrics(pwv_pred, pwv_true_t)
    m_trans_pi = compute_metrics(pi_pred, pwv_true_t / zwd_t)

    print('\nProfileTransformer PWV:', flush=True)
    print(f'  RMSE={m_trans["RMSE"]:.4f}  MAE={m_trans["MAE"]:.4f}  '
          f'R2={m_trans["R2"]:.4f}  Bias={m_trans["Bias"]:.4f}', flush=True)
    print(f'  Pi RMSE={m_trans_pi["RMSE"]:.6f}  Pi R2={m_trans_pi["R2"]:.4f}', flush=True)

    # --- 对比汇总 ---
    os.makedirs(args.output_dir, exist_ok=True)

    summary = pd.DataFrame([
        {'Method': 'Saastamoinen (full)', **m_saast_full},
        {'Method': 'Saastamoinen (Pi-only)', **m_saast_pi},
        {'Method': 'ProfileTransformer', **m_trans},
    ])
    summary_path = os.path.join(args.output_dir, 'comparison_metrics.csv')
    summary.to_csv(summary_path, index=False, float_format='%.6f')
    print(f'\n对比指标已保存: {summary_path}', flush=True)

    print('\n' + '=' * 70, flush=True)
    print(f'{"方法":<25} {"RMSE(mm)":<10} {"MAE(mm)":<10} {"R2":<8} {"Bias(mm)":<10}', flush=True)
    print('-' * 70, flush=True)
    for _, r in summary.iterrows():
        print(f'{r["Method"]:<25} {r["RMSE"]:<10.4f} {r["MAE"]:<10.4f} {r["R2"]:<8.4f} {r["Bias"]:<10.4f}', flush=True)
    print('=' * 70, flush=True)

    improvement = (m_saast_pi['RMSE'] - m_trans['RMSE']) / m_saast_pi['RMSE'] * 100
    print(f'Transformer 相比 Saastamoinen(Pi-only) RMSE 改善: {improvement:.1f}%', flush=True)

    # --- 对比散点图 ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for ax, (pred, title, color) in zip(axes, [
        (pwv_saast_pi, 'Saastamoinen (Pi-only)', 'coral'),
        (pwv_saast_full, 'Saastamoinen (full)', 'goldenrod'),
        (pwv_pred, 'ProfileTransformer', 'steelblue'),
    ]):
        true_ref = pwv_true if title != 'ProfileTransformer' else pwv_true_t
        ax.scatter(true_ref, pred, s=4, alpha=0.25, c=color)
        vmin = float(min(true_ref.min(), pred.min()))
        vmax = float(max(true_ref.max(), pred.max()))
        ax.plot([vmin, vmax], [vmin, vmax], 'r--', linewidth=1)
        ax.set_xlabel('True PWV (mm)')
        ax.set_ylabel('Predicted PWV (mm)')
        m = compute_metrics(pred, true_ref)
        ax.set_title(f'{title}\nRMSE={m["RMSE"]:.3f} R2={m["R2"]:.4f}')
        ax.grid(alpha=0.3)
    plt.tight_layout()
    scatter_path = os.path.join(args.output_dir, 'comparison_scatter.png')
    plt.savefig(scatter_path, dpi=300)
    plt.close()
    print(f'对比散点图已保存: {scatter_path}', flush=True)

    # --- 误差分布直方图 ---
    fig, ax = plt.subplots(figsize=(8, 5))
    err_saast = pwv_saast_pi - pwv_true
    err_trans = pwv_pred - pwv_true_t
    bins = np.linspace(-5, 5, 50)
    ax.hist(err_saast, bins=bins, alpha=0.5, label='Saastamoinen (Pi-only)', color='coral')
    ax.hist(err_trans, bins=bins, alpha=0.5, label='ProfileTransformer', color='steelblue')
    ax.set_xlabel('PWV Error (mm)')
    ax.set_ylabel('Count')
    ax.set_title('Error Distribution')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    err_path = os.path.join(args.output_dir, 'error_distribution.png')
    plt.savefig(err_path, dpi=300)
    plt.close()
    print(f'误差分布图已保存: {err_path}', flush=True)

    # --- 逐站点指标 ---
    per_station = []
    for sid in sorted(set(stations_t)):
        idx = [i for i, s in enumerate(stations_t) if s == sid]
        m = compute_metrics(pwv_pred[idx], pwv_true_t[idx])
        m['station_id'] = sid
        per_station.append(m)
    per_station_df = pd.DataFrame(per_station)
    per_station_path = os.path.join(args.output_dir, 'per_station_metrics.csv')
    per_station_df.to_csv(per_station_path, index=False, float_format='%.6f')
    print(f'逐站点指标已保存: {per_station_path}', flush=True)

    print('评估完成!', flush=True)


def main():
    parser = argparse.ArgumentParser(description='Transformer vs Saastamoinen 对比评估')
    parser.add_argument('--data_dir', type=str, default=r'D:\gnss水汽反演\第一阶段\xg_test')
    parser.add_argument('--model_path', type=str, default='result/best_model.pth')
    parser.add_argument('--output_dir', type=str, default='result')
    parser.add_argument('--test_station_ratio', type=float, default=0.1)
    parser.add_argument('--val_ratio', type=float, default=0.15)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max_files', type=int, default=None)
    parser.add_argument('--test_stations', type=str, default=None,
                        help='指定测试站列表文件路径(与训练一致)')
    args = parser.parse_args()

    if not os.path.isabs(args.model_path):
        args.model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.model_path)
    if not os.path.isabs(args.output_dir):
        args.output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.output_dir)

    evaluate(args)


if __name__ == '__main__':
    main()