# -*- coding: utf-8 -*-
"""
第二阶段训练脚本: ProfileTransformer ZWD -> PWV 直接映射模型

训练目标:
  - 模型输出转换系数 Pi, PWV = Pi * ZWD
  - 损失函数: SmoothL1 (Huber) on PWV, 对异常值鲁棒
  - 评估指标: RMSE, MAE, R2, Bias on PWV
  - 早停: 验证集 RMSE 连续 patience 轮不下降则停止

用法:
  本地测试:
    python train.py --data_dir D:/gnss水汽反演/第一阶段/xg_test --output_dir result

  服务器全量训练:
    python train.py --data_dir /path/to/train_data --output_dir result --batch_size 128 --epochs 100
"""
import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 同目录导入
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import prepare_data, collate_profiles, GLOBAL_FEATURE_DIM
from model import ProfileTransformer

import warnings
warnings.filterwarnings('ignore')

# 环境变量
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_metrics(pred, true):
    pred = np.asarray(pred, dtype=np.float64).flatten()
    true = np.asarray(true, dtype=np.float64).flatten()
    rmse = float(np.sqrt(mean_squared_error(true, pred)))
    mae = float(mean_absolute_error(true, pred))
    r2 = float(r2_score(true, pred))
    bias = float(np.mean(pred - true))
    rel = rmse / np.mean(true) * 100 if np.mean(true) > 0 else 0.0
    return {'RMSE': rmse, 'MAE': mae, 'R2': r2, 'Bias': bias, 'Rel_RMSE_pct': rel, 'N': len(true)}


def evaluate_model(model, loader, device, zwd_input=True):
    """在给定 loader 上评估, 返回指标和预测值."""
    model.eval()
    all_pwv_pred, all_pwv_true, all_pi_pred, all_pi_true = [], [], [], []
    all_zwd, all_station_ids, all_times, all_lats, all_lons, all_elvs, all_tms = [], [], [], [], [], [], []

    with torch.no_grad():
        for batch in loader:
            levels = batch['levels'].to(device)
            heights = batch['heights'].to(device)
            global_feat = batch['global_feat'].to(device)
            mask = batch['attention_mask'].to(device)
            zwd = batch['zwd'].to(device)
            pi_true = batch['pi'].to(device)
            pwv_true = batch['pwv'].to(device)

            pi_pred = model(levels, heights, global_feat, mask)
            pwv_pred = pi_pred * zwd

            all_pwv_pred.extend(pwv_pred.cpu().numpy())
            all_pwv_true.extend(pwv_true.cpu().numpy())
            all_pi_pred.extend(pi_pred.cpu().numpy())
            all_pi_true.extend(pi_true.cpu().numpy())
            all_zwd.extend(zwd.cpu().numpy())
            all_station_ids.extend(batch['station_ids'])
            all_times.extend(batch['times'])
            all_lats.extend(batch['lats'].numpy())
            all_lons.extend(batch['lons'].numpy())
            all_elvs.extend(batch['elvs'].numpy())
            all_tms.extend(batch['tms'].numpy())

    pwv_m = compute_metrics(all_pwv_pred, all_pwv_true)
    pi_m = compute_metrics(all_pi_pred, all_pi_true)
    return (pwv_m, pi_m, np.array(all_pwv_pred), np.array(all_pwv_true),
            np.array(all_pi_pred), np.array(all_pi_true), np.array(all_zwd),
            all_station_ids, all_times, np.array(all_lats), np.array(all_lons),
            np.array(all_elvs), np.array(all_tms))


def plot_scatter(true, pred, save_path, title, color='steelblue'):
    true = np.asarray(true).flatten()
    pred = np.asarray(pred).flatten()
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(true, pred, s=4, alpha=0.25, c=color)
    vmin = float(min(true.min(), pred.min()))
    vmax = float(max(true.max(), pred.max()))
    ax.plot([vmin, vmax], [vmin, vmax], 'r--', linewidth=1)
    ax.set_xlabel('True PWV (mm)')
    ax.set_ylabel('Predicted PWV (mm)')
    ax.set_title(title)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()


def train(args):
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}', flush=True)

    # 数据准备
    train_loader, val_loader, test_loader, scalers, info = prepare_data(
        args.data_dir,
        batch_size=args.batch_size,
        max_len=args.max_len,
        test_station_ratio=args.test_station_ratio,
        val_ratio=args.val_ratio,
        random_state=args.seed,
        max_files=args.max_files,
        num_workers=args.num_workers,
        test_stations=args.test_stations,
    )
    print(f'数据: train={info["n_train"]} val={info["n_val"]} test={info["n_test"]}', flush=True)

    # 模型
    model = ProfileTransformer(
        level_feat_dim=4,
        global_feat_dim=GLOBAL_FEATURE_DIM,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        ff_dim=args.ff_dim,
        dropout=args.dropout,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'模型参数量: {n_params:,}', flush=True)

    # 优化器 & 调度器
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)
    criterion = nn.SmoothL1Loss(beta=1.0)  # Huber loss, beta=1.0 时对 |err|<1 是 L2, 对 |err|>1 是 L1

    # 训练循环
    best_val_rmse = float('inf')
    patience_counter = 0
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # 保存 scalers 供 evaluate.py 使用 (不放进 torch checkpoint, 避免 weights_only 限制)
    import pickle
    with open(os.path.join(output_dir, 'scalers.pkl'), 'wb') as f:
        pickle.dump(scalers, f)
    print(f'scalers 已保存: {os.path.join(output_dir, "scalers.pkl")}', flush=True)

    history = {'train_loss': [], 'val_rmse': [], 'val_r2': []}

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

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

            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = epoch_loss / max(n_batches, 1)
        history['train_loss'].append(avg_loss)

        # 验证
        if val_loader is not None:
            val_m, val_pi_m, _, _, _, _, _, _, _, _, _, _, _ = evaluate_model(model, val_loader, device)
            history['val_rmse'].append(val_m['RMSE'])
            history['val_r2'].append(val_m['R2'])

            log = f'Epoch {epoch:3d}/{args.epochs}  loss={avg_loss:.4f}  val_RMSE={val_m["RMSE"]:.4f}  val_R2={val_m["R2"]:.4f}  lr={scheduler.get_last_lr()[0]:.2e}'

            if val_m['RMSE'] < best_val_rmse:
                best_val_rmse = val_m['RMSE']
                patience_counter = 0
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'epoch': epoch,
                    'val_rmse': val_m['RMSE'],
                    'args': vars(args),
                }, os.path.join(output_dir, 'best_model.pth'))
                log += '  *'
            else:
                patience_counter += 1

            print(log, flush=True)

            if patience_counter >= args.patience:
                print(f'早停: 验证 RMSE 连续 {args.patience} 轮未下降', flush=True)
                break
        else:
            print(f'Epoch {epoch:3d}/{args.epochs}  loss={avg_loss:.4f}  lr={scheduler.get_last_lr()[0]:.2e}', flush=True)
            if epoch % 5 == 0 or epoch == args.epochs:
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'epoch': epoch,
                    'args': vars(args),
                }, os.path.join(output_dir, 'best_model.pth'))

    # 加载最佳模型, 在测试集上评估
    ckpt_path = os.path.join(output_dir, 'best_model.pth')
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
        model.load_state_dict(ckpt['model_state_dict'])
        print(f'已加载最佳模型 (epoch {ckpt.get("epoch", "?")})', flush=True)

    test_m, test_pi_m, pwv_pred, pwv_true, pi_pred, pi_true, zwd_arr, station_ids, \
        times, lats, lons, elvs, tms = evaluate_model(model, test_loader, device)

    print('=' * 60, flush=True)
    print('测试集 PWV 评估结果', flush=True)
    print(f'  样本数:   {test_m["N"]}', flush=True)
    print(f'  RMSE:     {test_m["RMSE"]:.4f} mm', flush=True)
    print(f'  MAE:      {test_m["MAE"]:.4f} mm', flush=True)
    print(f'  R2:       {test_m["R2"]:.4f}', flush=True)
    print(f'  Bias:     {test_m["Bias"]:.4f} mm', flush=True)
    print(f'  Rel RMSE: {test_m["Rel_RMSE_pct"]:.2f}%', flush=True)
    print(f'  Pi RMSE:  {test_pi_m["RMSE"]:.6f}  Pi R2: {test_pi_m["R2"]:.4f}', flush=True)
    print('=' * 60, flush=True)

    # 保存预测结果
    import pandas as pd
    result_df = pd.DataFrame({
        'station_id': station_ids,
        'time': times,
        'lat': lats,
        'lon': lons,
        'elv': elvs,
        'tm': tms,
        'zwd': zwd_arr,
        'pwv_true': pwv_true,
        'pwv_pred': pwv_pred,
        'pwv_error': pwv_pred - pwv_true,
        'pi_true': pi_true,
        'pi_pred': pi_pred,
    })
    result_csv = os.path.join(output_dir, 'test_predictions.csv')
    result_df.to_csv(result_csv, index=False, float_format='%.6f')
    print(f'预测结果已保存: {result_csv}', flush=True)

    # 散点图
    plot_scatter(pwv_true, pwv_pred, os.path.join(output_dir, 'scatter_pwv_transformer.png'),
                 'ProfileTransformer PWV: Predicted vs True')

    # 逐站点指标
    per_station = []
    for sid in sorted(set(station_ids)):
        idx = [i for i, s in enumerate(station_ids) if s == sid]
        m = compute_metrics(pwv_pred[idx], pwv_true[idx])
        m['station_id'] = sid
        per_station.append(m)
    per_station_df = pd.DataFrame(per_station)
    per_station_path = os.path.join(output_dir, 'per_station_metrics.csv')
    per_station_df.to_csv(per_station_path, index=False, float_format='%.6f')
    print(f'逐站点指标已保存: {per_station_path}', flush=True)

    # 保存指标
    with open(os.path.join(output_dir, 'metrics.txt'), 'w', encoding='utf-8') as f:
        f.write('ProfileTransformer PWV 测试集评估结果\n')
        f.write('=' * 60 + '\n')
        f.write(f'样本数:   {test_m["N"]}\n')
        f.write(f'RMSE:     {test_m["RMSE"]:.4f} mm\n')
        f.write(f'MAE:      {test_m["MAE"]:.4f} mm\n')
        f.write(f'R2:       {test_m["R2"]:.4f}\n')
        f.write(f'Bias:     {test_m["Bias"]:.4f} mm\n')
        f.write(f'Rel RMSE: {test_m["Rel_RMSE_pct"]:.2f}%\n')
        f.write(f'\nPi 评估:\n')
        f.write(f'Pi RMSE:  {test_pi_m["RMSE"]:.6f}\n')
        f.write(f'Pi R2:    {test_pi_m["R2"]:.4f}\n')
        f.write(f'Pi Bias:   {test_pi_m["Bias"]:.6f}\n')
        f.write(f'\n模型超参数:\n')
        for k, v in vars(args).items():
            f.write(f'  {k}: {v}\n')

    # 保存训练曲线
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    if history['train_loss']:
        axes[0].plot(history['train_loss'])
        axes[0].set_title('Training Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].grid(alpha=0.3)
    if history['val_rmse']:
        axes[1].plot(history['val_rmse'])
        axes[1].set_title('Validation RMSE')
        axes[1].set_xlabel('Epoch')
        axes[1].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_curve.png'), dpi=200)
    plt.close()

    print('训练完成!', flush=True)
    return test_m


def main():
    parser = argparse.ArgumentParser(description='ProfileTransformer 训练')
    parser.add_argument('--data_dir', type=str, default=r'D:/gnss水汽反演/第一阶段/xg_test',
                        help='数据目录')
    parser.add_argument('--output_dir', type=str, default='result',
                        help='输出目录')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--max_len', type=int, default=30, help='廓线最大长度')
    parser.add_argument('--epochs', type=int, default=80)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--d_model', type=int, default=128)
    parser.add_argument('--n_heads', type=int, default=8)
    parser.add_argument('--n_layers', type=int, default=4)
    parser.add_argument('--ff_dim', type=int, default=512)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--test_station_ratio', type=float, default=0.1)
    parser.add_argument('--val_ratio', type=float, default=0.15)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max_files', type=int, default=None, help='最多读取文件数(本地测试)')
    parser.add_argument('--test_stations', type=str, default=None,
                        help='指定测试站列表文件路径(每行一个站ID), 与第一阶段36测试站对齐时使用')
    parser.add_argument('--num_workers', type=int, default=0)
    args = parser.parse_args()

    output_dir = args.output_dir
    if not os.path.isabs(output_dir):
        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), output_dir)
    args.output_dir = output_dir

    print(f'数据目录: {args.data_dir}', flush=True)
    print(f'输出目录: {args.output_dir}', flush=True)
    train(args)


if __name__ == '__main__':
    main()