# -*- coding: utf-8 -*-
"""
第二阶段: 留一站交叉验证 (Leave-One-Station-Out, LOSO) 评估

背景:
  本地只有 10 个探空站, 若按 test_station_ratio=0.1 只留 1 个站做测试,
  单次结果受该站气候特征影响极大, 无法代表模型真实水平.
  LOSO 轮流留出每个站作为测试, 用其余 9 个站训练, 聚合 10 折结果,
  得到更稳健的本地估计, 并与 Saastamoinen 基线(全流程 / Pi-only)对比.

用法:
  python loso_evaluate.py --data_dir D:/gnss水汽反演/第一阶段/xg_test --output_dir result_loso
  # 完整训练(服务器): 数据量大时用 --epochs 80 --patience 15
"""
import os
import sys
import glob
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)
sys.path.insert(0, os.path.dirname(script_dir))  # 父目录(含 saastamoinen_pwv.py)

from data import load_all_profiles, ProfileDataset, collate_profiles, _encode_global, GLOBAL_FEATURE_DIM
from model import ProfileTransformer
from train import compute_metrics, set_seed, evaluate_model
from saastamoinen_pwv import saastamoinen_pwv_batch

# 物理常数
K2P = 22.1
K3 = 3.739e5
RV = 461.495
RHO_W = 1000.0


def saastamoinen_pi(Tm_K):
    return 1e8 / (RHO_W * RV * (K3 / Tm_K + K2P))


def make_loaders(train_profiles, val_profiles, test_profiles, batch_size, max_len,
                num_workers=0, scalers=None):
    """构建 train/val/test DataLoader.

    scalers 为 None 时按训练数据拟合; 否则(测试阶段)直接使用传入的 scalers.
    """
    from torch.utils.data import DataLoader

    if scalers is None:
        train_ds = ProfileDataset(train_profiles, fit_scalers=True)
        scalers = train_ds.get_scalers()
    train_ds = ProfileDataset(train_profiles, **scalers) if len(train_profiles) > 0 else None
    val_ds = ProfileDataset(val_profiles, **scalers) if len(val_profiles) > 0 else None
    test_ds = ProfileDataset(test_profiles, **scalers)

    collate_fn = lambda b: collate_profiles(b, max_len=max_len)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              collate_fn=collate_fn, num_workers=num_workers, pin_memory=True) if train_ds is not None else None
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            collate_fn=collate_fn, num_workers=num_workers, pin_memory=True) if val_ds is not None else None
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             collate_fn=collate_fn, num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader, test_loader, scalers


def train_fold(train_profiles, val_profiles, args, device):
    """训练一个折的 ProfileTransformer, 返回 (model, scalers)."""
    set_seed(args.seed)
    train_loader, val_loader, _, scalers = make_loaders(
        train_profiles, val_profiles, [], args.batch_size, args.max_len, args.num_workers)

    model = ProfileTransformer(
        level_feat_dim=4, global_feat_dim=GLOBAL_FEATURE_DIM,
        d_model=args.d_model, n_heads=args.n_heads, n_layers=args.n_layers,
        ff_dim=args.ff_dim, dropout=args.dropout,
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)
    criterion = nn.SmoothL1Loss(beta=1.0)

    best_val_rmse = float('inf')
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        for batch in train_loader:
            levels = batch['levels'].to(device)
            heights = batch['heights'].to(device)
            global_feat = batch['global_feat'].to(device)
            mask = batch['attention_mask'].to(device)
            zwd = batch['zwd'].to(device)
            pwv_true = batch['pwv'].to(device)

            pi_pred = model(levels, heights, global_feat, mask)
            pwv_pred = pi_pred * zwd
            loss = criterion(pwv_pred, pwv_true)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        scheduler.step()

        if val_loader is not None:
            val_m, _, _, _, _, _, _, _ = evaluate_model(model, val_loader, device)
            if val_m['RMSE'] < best_val_rmse:
                best_val_rmse = val_m['RMSE']
                patience_counter = 0
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            else:
                patience_counter += 1
            if patience_counter >= args.patience:
                break
        else:
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    if val_loader is not None and 'best_state' in dir():
        model.load_state_dict(best_state)
    return model, scalers


def predict_loader(model, loader, device):
    """对 loader 推理, 返回 (pwv_pred, pwv_true, pi_pred, zwd, station_ids)."""
    model.eval()
    all_pred, all_true, all_pi, all_zwd, all_sid = [], [], [], [], []
    with torch.no_grad():
        for batch in loader:
            levels = batch['levels'].to(device)
            heights = batch['heights'].to(device)
            global_feat = batch['global_feat'].to(device)
            mask = batch['attention_mask'].to(device)
            zwd = batch['zwd'].to(device)
            pi_pred = model(levels, heights, global_feat, mask)
            pwv_pred = pi_pred * zwd
            all_pred.extend(pwv_pred.cpu().numpy())
            all_true.extend(batch['pwv'].numpy())
            all_pi.extend(pi_pred.cpu().numpy())
            all_zwd.extend(zwd.cpu().numpy())
            all_sid.extend(batch['station_ids'])
    return (np.array(all_pred), np.array(all_true), np.array(all_pi),
            np.array(all_zwd), all_sid)


def main():
    parser = argparse.ArgumentParser(description='LOSO 留一站交叉验证')
    parser.add_argument('--data_dir', type=str, default=r'D:/gnss水汽反演/第一阶段/xg_test')
    parser.add_argument('--output_dir', type=str, default='result_loso')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--patience', type=int, default=6)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--max_len', type=int, default=30)
    parser.add_argument('--val_ratio', type=float, default=0.15)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--d_model', type=int, default=128)
    parser.add_argument('--n_heads', type=int, default=8)
    parser.add_argument('--n_layers', type=int, default=4)
    parser.add_argument('--ff_dim', type=int, default=512)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--max_files', type=int, default=None)
    parser.add_argument('--test_stations', type=str, default=None,
                        help='仅对指定站点做折(逗号分隔), 用于快速消融实验')
    args = parser.parse_args()

    if not os.path.isabs(args.output_dir):
        args.output_dir = os.path.join(script_dir, args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'设备: {device}', flush=True)

    profiles = load_all_profiles(args.data_dir, max_files=args.max_files)
    if len(profiles) == 0:
        raise SystemExit('无数据!')

    # 需要 tm_surface 字段 -> 从 data.py 的 read_station_file 补上
    # 这里直接复用 read_station_file 重新读取(在文件末尾补充 tm_surface)
    from data import read_station_file
    stations = sorted(set(p['station_id'] for p in profiles))
    if args.test_stations:
        wanted = set(s.strip() for s in args.test_stations.split(','))
        stations = [s for s in stations if s in wanted]
    print(f'LOSO 共 {len(stations)} 个站, 每站轮流作为测试站', flush=True)

    all_records = []      # Transformer 预测汇总
    fold_rows = []        # 每折指标

    for fold_i, test_station in enumerate(stations, 1):
        print(f'\n===== Fold {fold_i}/{len(stations)}: 测试站 {test_station} =====', flush=True)
        test_profiles = [p for p in profiles if p['station_id'] == test_station]
        train_val_profiles = [p for p in profiles if p['station_id'] != test_station]
        train_val_profiles.sort(key=lambda p: p['time_str'])
        n_val = int(len(train_val_profiles) * args.val_ratio)
        val_profiles = train_val_profiles[-n_val:] if n_val > 0 else []
        train_profiles = train_val_profiles[:-n_val] if n_val > 0 else train_val_profiles

        # 训练 Transformer
        model, scalers = train_fold(train_profiles, val_profiles, args, device)
        _, _, test_loader, _ = make_loaders(
            [], [], test_profiles, args.batch_size, args.max_len, args.num_workers, scalers=scalers)
        pwv_pred, pwv_true, pi_pred, zwd, sids = predict_loader(model, test_loader, device)
        m = compute_metrics(pwv_pred, pwv_true)
        fold_rows.append({
            'fold': fold_i, 'test_station': test_station, 'N': int(m['N']),
            'RMSE': m['RMSE'], 'MAE': m['MAE'], 'R2': m['R2'], 'Bias': m['Bias'],
        })
        for i in range(len(pwv_pred)):
            all_records.append({
                'station_id': sids[i], 'zwd': zwd[i], 'pwv_true': pwv_true[i],
                'pwv_pred': pwv_pred[i], 'pi_pred': pi_pred[i],
            })
        print(f'  折结果: RMSE={m["RMSE"]:.4f}  MAE={m["MAE"]:.4f}  '
              f'R2={m["R2"]:.4f}  Bias={m["Bias"]:.4f}', flush=True)

    # 汇总 Transformer
    df_t = pd.DataFrame(all_records)
    df_t['pwv_error'] = df_t['pwv_pred'] - df_t['pwv_true']
    m_t = compute_metrics(df_t['pwv_pred'].values, df_t['pwv_true'].values)

    # Saastamoinen 基线 (全部站点, 与 Transformer 测试集一致)
    base_records = []
    for p in profiles:
        surface = p['levels'][0]
        base_records.append({
            'station_id': p['station_id'], 'time_str': p['time_str'],
            'lat': p['global_raw']['lat'], 'elv': float(surface[0]),
            'ps': float(surface[2]), 'ts': float(surface[1]), 'wps': float(surface[3]),
            'zwd': p['zwd_surface'], 'pwv': p['pwv_surface'],
            'tm': p.get('tm_surface', np.nan),
        })
    df_b = pd.DataFrame(base_records).dropna(subset=['tm'])
    df_b['pi_saast'] = saastamoinen_pi(df_b['tm'].values)
    df_b['pwv_saast_pi'] = df_b['pi_saast'] * df_b['zwd']
    df_b['pwv_saast_full'] = saastamoinen_pwv_batch(
        df_b['lat'].values, df_b['elv'].values, df_b['ps'].values,
        df_b['wps'].values, df_b['ts'].values, df_b['tm'].values)

    m_full = compute_metrics(df_b['pwv_saast_full'].values, df_b['pwv'].values)
    m_pi = compute_metrics(df_b['pwv_saast_pi'].values, df_b['pwv'].values)

    # 汇总表
    summary = pd.DataFrame([
        {'Method': 'Saastamoinen (full)', **m_full},
        {'Method': 'Saastamoinen (Pi-only)', **m_pi},
        {'Method': 'ProfileTransformer (LOSO)', **m_t},
    ])
    summary.to_csv(os.path.join(args.output_dir, 'comparison_metrics.csv'), index=False, float_format='%.6f')
    fold_df = pd.DataFrame(fold_rows)
    fold_df.to_csv(os.path.join(args.output_dir, 'per_fold_metrics.csv'), index=False, float_format='%.6f')
    df_t.to_csv(os.path.join(args.output_dir, 'loso_predictions.csv'), index=False, float_format='%.6f')

    print('\n' + '=' * 78)
    print(f'{"方法":<28} {"RMSE(mm)":<10} {"MAE(mm)":<10} {"R2":<8} {"Bias(mm)":<10}')
    print('-' * 78)
    for _, r in summary.iterrows():
        print(f'{r["Method"]:<28} {r["RMSE"]:<10.4f} {r["MAE"]:<10.4f} {r["R2"]:<8.4f} {r["Bias"]:<10.4f}')
    print('=' * 78)
    if m_pi['RMSE'] > 0:
        print(f'Transformer 相比 Saastamoinen(Pi-only) RMSE 改善: {(m_pi["RMSE"]-m_t["RMSE"])/m_pi["RMSE"]*100:.1f}%')

    # 图: 对比散点
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for ax, (pred, title, color) in zip(axes, [
        (df_b['pwv_saast_pi'].values, 'Saastamoinen (Pi-only)', 'coral'),
        (df_b['pwv_saast_full'].values, 'Saastamoinen (full)', 'goldenrod'),
        (df_t['pwv_pred'].values, 'ProfileTransformer (LOSO)', 'steelblue'),
    ]):
        true_ref = df_b['pwv'].values if title != 'ProfileTransformer (LOSO)' else df_t['pwv_true'].values
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
    plt.savefig(os.path.join(args.output_dir, 'comparison_scatter.png'), dpi=300)
    plt.close()

    # 图: 误差分布
    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.linspace(-6, 6, 60)
    ax.hist(df_b['pwv_saast_pi'].values - df_b['pwv'].values, bins=bins, alpha=0.5,
            label='Saastamoinen (Pi-only)', color='coral')
    ax.hist(df_t['pwv_error'].values, bins=bins, alpha=0.5,
            label='ProfileTransformer (LOSO)', color='steelblue')
    ax.set_xlabel('PWV Error (mm)')
    ax.set_ylabel('Count')
    ax.set_title('Error Distribution (LOSO)')
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'error_distribution.png'), dpi=300)
    plt.close()

    # 图: 逐折 RMSE
    fig, ax = plt.subplots(figsize=(10, 4))
    x = np.arange(len(fold_df))
    ax.bar(x, fold_df['RMSE'].values, color='steelblue')
    ax.axhline(m_pi['RMSE'], color='coral', linestyle='--', label=f"Saastamoinen Pi-only RMSE={m_pi['RMSE']:.3f}")
    ax.set_xticks(x)
    ax.set_xticklabels(fold_df['test_station'].values, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('RMSE (mm)')
    ax.set_title('Per-fold Test Station RMSE (LOSO)')
    ax.legend(); ax.grid(alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'per_fold_rmse.png'), dpi=300)
    plt.close()

    print(f'\n结果已保存到: {args.output_dir}')


if __name__ == '__main__':
    main()
