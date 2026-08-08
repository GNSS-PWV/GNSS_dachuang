# -*- coding: utf-8 -*-
"""
消融实验: ProfileTransformer 各设计环节
变体:
  baseline   : 完整模型 (廓线 + 全局特征, Pi参数化)
  no_global  : 去掉全局特征(只廓线)
  no_profile : 去掉廓线(只全局特征 -> MLP)
  direct_pwv : 直接预测 PWV (不做 Pi 参数化)
  layers2    : 编码器 2 层
  layers6    : 编码器 6 层
用法:
  python ablation_train.py --data_dir <xg_data> --variant baseline --out result_ablation
"""
import os, sys, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from data import prepare_data, GLOBAL_FEATURE_DIM
from model import ProfileTransformer
from train import set_seed


class GlobalOnlyModel(nn.Module):
    """消融: 只用全局特征(无廓线) -> MLP -> Pi"""
    def __init__(self, global_feat_dim, hidden=128, dropout=0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(global_feat_dim, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )
        self.pi_min, self.pi_max = 0.05, 0.35

    def forward(self, gf):
        return self.pi_min + torch.sigmoid(self.mlp(gf).squeeze(-1)) * (self.pi_max - self.pi_min)


def build_model(variant, args):
    if variant == 'no_profile':
        return GlobalOnlyModel(GLOBAL_FEATURE_DIM, hidden=args.d_model, dropout=args.dropout)
    return ProfileTransformer(
        level_feat_dim=4, global_feat_dim=GLOBAL_FEATURE_DIM,
        d_model=args.d_model, n_heads=args.n_heads, n_layers=args.n_layers,
        ff_dim=args.ff_dim, dropout=args.dropout,
        use_global=(variant != 'no_global'),
        direct_pwv=(variant == 'direct_pwv'))


def predict(model, variant, levels, heights, gf, mask, zwd):
    if variant == 'no_profile':
        return model(gf)
    out = model(levels, heights, gf, mask)
    if variant == 'direct_pwv':
        return out
    return out * zwd


def metrics(pred, true):
    pred = np.asarray(pred); true = np.asarray(true)
    return {'RMSE': float(np.sqrt(mean_squared_error(true, pred))),
            'MAE': float(mean_absolute_error(true, pred)),
            'R2': float(r2_score(true, pred)),
            'Bias': float(np.mean(pred - true)), 'N': len(pred)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', required=True)
    ap.add_argument('--variant', required=True, choices=['baseline', 'no_global', 'no_profile',
                                                         'direct_pwv', 'layers2', 'layers6'])
    ap.add_argument('--out', default='result_ablation')
    ap.add_argument('--max_files', type=int, default=200)
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--batch_size', type=int, default=128)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--patience', type=int, default=10)
    ap.add_argument('--d_model', type=int, default=128)
    ap.add_argument('--n_heads', type=int, default=8)
    ap.add_argument('--n_layers', type=int, default=4)
    ap.add_argument('--ff_dim', type=int, default=512)
    ap.add_argument('--dropout', type=float, default=0.1)
    ap.add_argument('--max_len', type=int, default=30)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--num_workers', type=int, default=4)
    ap.add_argument('--test_stations', default=None)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'变体: {args.variant}  设备: {device}', flush=True)

    train_loader, val_loader, test_loader, scalers, info = prepare_data(
        args.data_dir, batch_size=args.batch_size, max_len=args.max_len,
        test_station_ratio=0.1, val_ratio=0.15, random_state=args.seed,
        max_files=args.max_files, num_workers=args.num_workers,
        test_stations=args.test_stations)
    print(f'数据: train={info["n_train"]} val={info["n_val"]} test={info["n_test"]}', flush=True)

    model = build_model(args.variant, args).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'参数量: {n_params:,}', flush=True)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr*0.01)
    criterion = nn.SmoothL1Loss(beta=1.0)

    best_val = float('inf'); best_state = None; patience = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        for batch in train_loader:
            levels = batch['levels'].to(device); heights = batch['heights'].to(device)
            gf = batch['global_feat'].to(device); mask = batch['attention_mask'].to(device)
            zwd = batch['zwd'].to(device); pwv_true = batch['pwv'].to(device)
            pwv_pred = predict(model, args.variant, levels, heights, gf, mask, zwd)
            loss = criterion(pwv_pred, pwv_true)
            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()
        if val_loader is not None:
            model.eval()
            vp, vt = [], []
            with torch.no_grad():
                for batch in val_loader:
                    levels = batch['levels'].to(device); heights = batch['heights'].to(device)
                    gf = batch['global_feat'].to(device); mask = batch['attention_mask'].to(device)
                    zwd = batch['zwd'].to(device)
                    vp.append(predict(model, args.variant, levels, heights, gf, mask, zwd).cpu().numpy())
                    vt.append(batch['pwv'].numpy())
            vp = np.concatenate(vp); vt = np.concatenate(vt)
            vr = float(np.sqrt(mean_squared_error(vt, vp)))
            if vr < best_val:
                best_val = vr; best_state = {k: v.clone() for k, v in model.state_dict().items()}; patience = 0
            else:
                patience += 1
            print(f'Epoch {epoch:3d}/{args.epochs}  loss={loss.item():.4f}  val_RMSE={vr:.4f}', flush=True)
            if patience >= args.patience:
                print('早停', flush=True); break
    if best_state:
        model.load_state_dict(best_state)

    # 测试评估
    model.eval()
    tp, tt = [], []
    with torch.no_grad():
        for batch in test_loader:
            levels = batch['levels'].to(device); heights = batch['heights'].to(device)
            gf = batch['global_feat'].to(device); mask = batch['attention_mask'].to(device)
            zwd = batch['zwd'].to(device)
            tp.append(predict(model, args.variant, levels, heights, gf, mask, zwd).cpu().numpy())
            tt.append(batch['pwv'].numpy())
    tp = np.concatenate(tp); tt = np.concatenate(tt)
    m = metrics(tp, tt)
    print(f'\n[{args.variant}] 测试: RMSE={m["RMSE"]:.4f}  MAE={m["MAE"]:.4f}  R2={m["R2"]:.6f}  Bias={m["Bias"]:.4f}  N={m["N"]}', flush=True)
    import pandas as pd
    pd.DataFrame([{'variant': args.variant, **m}]).to_csv(
        os.path.join(args.out, f'metrics_{args.variant}.csv'), index=False, float_format='%.6f')
    print('完成', flush=True)


if __name__ == '__main__':
    main()
