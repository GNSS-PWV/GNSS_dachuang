# -*- coding: utf-8 -*-
"""
Optuna 超参搜索: ProfileTransformer (第三阶段模型优化预研)

策略: 用数据子集(max_files)快速试超参, 每 trial 训练固定轮数, 目标=验证集 RMSE 最小.
搜完后把最优超参打印/保存, 供全量重训使用.

用法(服务器):
  python optuna_search.py --data_dir <xg_data> --max_files 150 --n_trials 20 --epochs 10
"""
import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from data import prepare_data
from model import ProfileTransformer
from train import evaluate_model, set_seed, compute_metrics


from optuna.exceptions import TrialPruned


def objective(trial, args, device):
    d_model = trial.suggest_categorical('d_model', [64, 128, 256])
    n_layers = trial.suggest_int('n_layers', 2, 6)
    n_heads = trial.suggest_categorical('n_heads', [4, 8, 16])
    ff_dim = trial.suggest_categorical('ff_dim', [256, 512, 1024])
    dropout = trial.suggest_float('dropout', 0.05, 0.3)
    lr = trial.suggest_float('lr', 3e-5, 5e-4, log=True)
    batch_size = trial.suggest_categorical('batch_size', [64, 128, 256])
    weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-3, log=True)
    # n_heads 需整除 d_model
    if d_model % n_heads != 0:
        raise trial.suggest_float('_invalid', 0, 0)

    train_loader, val_loader, test_loader, scalers, info = prepare_data(
        args.data_dir, batch_size=batch_size, max_len=args.max_len,
        test_station_ratio=0.1, val_ratio=0.15, random_state=args.seed,
        max_files=args.max_files, num_workers=args.num_workers,
        test_stations=args.test_stations)
    n_train = info['n_train']
    if n_train == 0:
        return float('inf')

    model = ProfileTransformer(level_feat_dim=4, global_feat_dim=9,
                               d_model=d_model, n_heads=n_heads, n_layers=n_layers,
                               ff_dim=ff_dim, dropout=dropout).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=lr * 0.01)
    criterion = nn.SmoothL1Loss(beta=1.0)

    best = float('inf')
    for epoch in range(1, args.epochs + 1):
        model.train()
        for batch in train_loader:
            levels = batch['levels'].to(device); heights = batch['heights'].to(device)
            gf = batch['global_feat'].to(device); mask = batch['attention_mask'].to(device)
            zwd = batch['zwd'].to(device); pwv_true = batch['pwv'].to(device)
            pwv_pred = model(levels, heights, gf, mask) * zwd
            loss = criterion(pwv_pred, pwv_true)
            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()
        if val_loader is not None:
            val_m, _, _, _, _, _, _, _, _, _, _, _, _ = evaluate_model(model, val_loader, device)
            best = min(best, val_m['RMSE'])
            trial.report(val_m['RMSE'], epoch)
            if trial.should_prune():
                raise TrialPruned()
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', required=True)
    ap.add_argument('--out', default='result_optuna')
    ap.add_argument('--n_trials', type=int, default=20)
    ap.add_argument('--epochs', type=int, default=10)
    ap.add_argument('--max_len', type=int, default=30)
    ap.add_argument('--max_files', type=int, default=150)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--num_workers', type=int, default=4)
    ap.add_argument('--test_stations', type=str, default=None)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    import optuna

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'设备: {device}', flush=True)

    def obj(trial):
        return objective(trial, args, device)

    study = optuna.create_study(direction='minimize',
                                storage='sqlite:///' + os.path.join(args.out, 'optuna.db'),
                                study_name='pwv_phase2', load_if_exists=True)
    study.optimize(obj, n_trials=args.n_trials, show_progress_bar=False)

    print('\n=== 最优超参 ===')
    print(study.best_params)
    print(f'最佳 val RMSE: {study.best_value:.4f} mm')
    with open(os.path.join(args.out, 'best_params.txt'), 'w', encoding='utf-8') as f:
        f.write(str(study.best_params) + '\n')
        f.write(f'best_val_rmse={study.best_value:.4f}\n')
    print(f'已保存 -> {args.out}')


if __name__ == '__main__':
    main()
