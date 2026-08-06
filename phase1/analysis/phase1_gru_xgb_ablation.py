# -*- coding: utf-8 -*-
"""
第一阶段紧凑消融: 纯GRU vs GRU+XGBoost残差修正 (PS 目标)

在数据子集上快速复现第一阶段的 GRU+XGB 思路, 量化 XGBoost 对 GRU 精度的提升.
特征: 时间周期(sin/cos) + 站点(lat/lon/elv) + 滞后/差分/滑动统计 (与第一阶段一致的精简版)
切分: 按站点 80/10/10 (训练/验证/测试), 测试站完全未见.
XGB: 以 [统计特征 + GRU预测] 预测残差, 加到 GRU 预测上.

输出: result_phase1_ablation/  (指标表 + 散点图)
"""
import os
import sys
import glob
import argparse
import numpy as np
import pandas as pd
import datetime as dt
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import RobustScaler

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size=96, num_layers=2, output_size=1, time_steps=6, dropout=0.3):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True,
                          dropout=dropout if num_layers > 1 else 0)
        self.ln = nn.LayerNorm(hidden_size)
        self.fc = nn.Sequential(nn.Linear(hidden_size, 64), nn.ReLU(), nn.Dropout(dropout),
                                nn.Linear(64, output_size))

    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc(self.ln(out[:, -1, :]))


def metrics(pred, true):
    pred = np.asarray(pred); true = np.asarray(true)
    rmse = float(np.sqrt(mean_squared_error(true, pred)))
    mae = float(mean_absolute_error(true, pred))
    r2 = float(r2_score(true, pred))
    mape = float(np.mean(np.abs((pred - true) / (true + 1e-8))) * 100)
    return {'RMSE': rmse, 'MAE': mae, 'R2': r2, 'MAPE': mape, 'N': len(true)}


def load_data(data_dir, max_files):
    files = sorted(glob.glob(os.path.join(data_dir, '**', '*_met.txt'), recursive=True))
    if not files:
        files = sorted(glob.glob(os.path.join(data_dir, '*_met.txt')))
    files = files[:max_files]
    dfs = []
    for fp in files:
        sid = os.path.basename(fp).split('_met')[0]
        try:
            df = pd.read_csv(fp, header=0, sep=None, engine='python')
            fc = df.columns[0]
            if str(fc).startswith('Unnamed') or str(fc).strip() == '':
                df = df.rename(columns={fc: 'TIME'})
            df['station_id'] = sid
            df['TIME'] = pd.to_datetime(df['TIME'], errors='coerce')
            dfs.append(df)
        except Exception:
            continue
    df = pd.concat(dfs, ignore_index=True)
    df = df.dropna(subset=['PS', 'TIME', 'ELV'])
    # 关键: 探空文件每个时刻含多个高度层(廓线), 只保留地表层(最小ELV)构成时间序列
    idx = df.groupby(['station_id', 'TIME'])['ELV'].idxmin()
    df = df.loc[idx].copy()
    df = df.sort_values(['station_id', 'TIME']).reset_index(drop=True)
    return df


def add_features(df, target='PS'):
    d = df.copy()
    d['year'] = d['TIME'].dt.year
    d['doy'] = d['TIME'].dt.dayofyear
    d['hour'] = d['TIME'].dt.hour + d['TIME'].dt.minute / 60
    d['month'] = d['TIME'].dt.month
    d['doy_sin'] = np.sin(2 * np.pi * d['doy'] / 365.25)
    d['doy_cos'] = np.cos(2 * np.pi * d['doy'] / 365.25)
    d['hour_sin'] = np.sin(2 * np.pi * d['hour'] / 24)
    d['hour_cos'] = np.cos(2 * np.pi * d['hour'] / 24)
    d['month_sin'] = np.sin(2 * np.pi * d['month'] / 12)
    d['month_cos'] = np.cos(2 * np.pi * d['month'] / 12)
    d['lat_sin'] = np.sin(np.deg2rad(d['LAT']))
    d['lat_cos'] = np.cos(np.deg2rad(d['LAT']))
    d['lon_sin'] = np.sin(np.deg2rad(d['LON']))
    d['lon_cos'] = np.cos(np.deg2rad(d['LON']))
    d['elv_norm'] = d['ELV'] / 5000.0
    # 滞后/差分/滑动统计 (同站内)
    g = d.groupby('station_id')[target]
    for lag in [1, 2, 3, 6, 12, 24]:
        d[f'{target}_lag{lag}'] = g.shift(lag)
    d[f'{target}_diff1'] = d[f'{target}'] - d[f'{target}_lag1']
    d[f'{target}_ma6'] = g.transform(lambda s: s.rolling(6, min_periods=1).mean())
    d[f'{target}_std6'] = g.transform(lambda s: s.rolling(6, min_periods=1).std())
    # 相邻观测间隔(小时), 用于判断时间连续性
    d['dt_h'] = g.diff()
    d['dt_h'] = d.groupby('station_id')['TIME'].diff().dt.total_seconds() / 3600
    d = d.dropna(subset=[f'{target}_lag{lag}' for lag in [1, 2, 3]])
    return d


def create_sequences(df, feat_cols, target, time_steps=6):
    X, y, meta = [], [], []
    for _, g in df.groupby('station_id'):
        vals = g[feat_cols].values
        ys = g[target].values
        for i in range(time_steps, len(g)):
            X.append(vals[i - time_steps:i])
            y.append(ys[i])
            meta.append((g['station_id'].iloc[i], g['TIME'].iloc[i]))
    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.float32), meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', required=True)
    ap.add_argument('--out', default='result_phase1_ablation')
    ap.add_argument('--max_files', type=int, default=80)
    ap.add_argument('--time_steps', type=int, default=12)
    ap.add_argument('--epochs', type=int, default=60)
    ap.add_argument('--batch_size', type=int, default=256)
    ap.add_argument('--test_ratio', type=float, default=0.2)
    ap.add_argument('--val_ratio', type=float, default=0.15)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    np.random.seed(args.seed); torch.manual_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'设备: {device}', flush=True)

    df = load_data(args.data_dir, args.max_files)
    print(f'原始数据: {len(df)} 行, {df["station_id"].nunique()} 站', flush=True)
    df = add_features(df, 'PS')
    print(f'加特征后: {len(df)} 行', flush=True)

    feat_cols = ['doy_sin', 'doy_cos', 'hour_sin', 'hour_cos', 'month_sin', 'month_cos',
                 'lat_sin', 'lat_cos', 'lon_sin', 'lon_cos', 'elv_norm',
                 'PS_lag1', 'PS_lag2', 'PS_lag3', 'PS_lag6', 'PS_lag12', 'PS_lag24',
                 'PS_diff1', 'PS_ma6', 'PS_std6', 'dt_h']
    feat_cols = [c for c in feat_cols if c in df.columns]
    # 关键: 去掉任何含 NaN 的特征行 (滚动std单值/diff首行会产生NaN)
    n_before = len(df)
    df = df.dropna(subset=feat_cols)
    print(f'特征行清理: {n_before} -> {len(df)} (去掉含NaN行)', flush=True)

    X, y, meta = create_sequences(df, feat_cols, 'PS', args.time_steps)
    print(f'序列: X={X.shape}', flush=True)

    # 按时间切分 (与第一阶段官方一致: 同一批站点, 按时间顺序预测未来)
    times = np.array([m[1] for m in meta], dtype='datetime64[ns]')
    order = np.argsort(times)
    n = len(order)
    n_test = int(n * args.test_ratio)
    n_val = int(n * args.val_ratio)
    te_idx = order[-n_test:] if n_test > 0 else np.array([], dtype=int)
    va_idx = order[-(n_test + n_val):-n_test] if (n_test + n_val) > 0 else np.array([], dtype=int)
    tr_idx = order[:-(n_test + n_val)] if (n_test + n_val) > 0 else order
    tr = np.zeros(n, dtype=bool); va = np.zeros(n, dtype=bool); te = np.zeros(n, dtype=bool)
    tr[tr_idx] = True; va[va_idx] = True; te[te_idx] = True
    print(f'切分(时间): 训练{tr.sum()} 验证{va.sum()} 测试{te.sum()}', flush=True)

    # 标准化 (注意: 先按序列掩码, 再展平到时间步)
    x_scaler = RobustScaler()
    x_scaler.fit(X[tr].reshape(-1, X.shape[2]))
    X_tr = x_scaler.transform(X[tr].reshape(-1, X.shape[2])).reshape(-1, args.time_steps, X.shape[2])
    X_va = x_scaler.transform(X[va].reshape(-1, X.shape[2])).reshape(-1, args.time_steps, X.shape[2])
    X_te = x_scaler.transform(X[te].reshape(-1, X.shape[2])).reshape(-1, args.time_steps, X.shape[2])
    y_scaler = RobustScaler()
    y_scaler.fit(y[tr].reshape(-1, 1))
    y_tr = y_scaler.transform(y[tr].reshape(-1, 1)).flatten()
    y_va = y_scaler.transform(y[va].reshape(-1, 1)).flatten()
    y_te_s = y_scaler.transform(y[te].reshape(-1, 1)).flatten()

    train_ds = TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr).unsqueeze(1))
    val_ds = TensorDataset(torch.from_numpy(X_va), torch.from_numpy(y_va).unsqueeze(1))
    tr_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    va_loader = DataLoader(val_ds, batch_size=args.batch_size)

    # 训练 GRU
    model = GRUModel(X.shape[2], hidden_size=96, num_layers=2, time_steps=args.time_steps).to(device)
    opt = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-5)
    crit = nn.SmoothL1Loss(beta=1.0)
    best_val = float('inf'); best_state = None
    for ep in range(args.epochs):
        model.train()
        for xb, yb in tr_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward(); opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            vp = []
            for xb, _ in va_loader:
                vp.append(model(xb.to(device)).cpu().numpy())
            vp = np.concatenate(vp).flatten()
        vr = float(np.sqrt(mean_squared_error(y_va, vp)))
        if vr < best_val:
            best_val = vr; best_state = {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    print(f'GRU 训练完成, 最佳验证 RMSE(标准化)={best_val:.4f}', flush=True)

    model.eval()
    with torch.no_grad():
        p_tr = model(torch.from_numpy(X_tr).to(device)).cpu().numpy().flatten()
        p_va = model(torch.from_numpy(X_va).to(device)).cpu().numpy().flatten()
        p_te = model(torch.from_numpy(X_te).to(device)).cpu().numpy().flatten()
    pred_tr = y_scaler.inverse_transform(p_tr.reshape(-1, 1)).flatten()
    pred_va = y_scaler.inverse_transform(p_va.reshape(-1, 1)).flatten()
    pred_te = y_scaler.inverse_transform(p_te.reshape(-1, 1)).flatten()
    ytr_o = y[tr]; yva_o = y[va]; yte_o = y[te]

    m_gru = metrics(pred_te, yte_o)
    print('\n=== 纯 GRU (测试集, 原始尺度 hPa) ===')
    print(f'  RMSE={m_gru["RMSE"]:.3f}  MAE={m_gru["MAE"]:.3f}  R2={m_gru["R2"]:.4f}  MAPE={m_gru["MAPE"]:.2f}%')

    # XGBoost 修正 (与官方 gru_ps.py 一致): 整段序列展平 + 均值/标准差 + GRU预测, 直接预测真值 PS (stacking)
    import xgboost as xgb
    def stack_feats(Xs, ps):
        flat = Xs.reshape(Xs.shape[0], -1)
        return np.hstack([flat,
                          flat.mean(axis=1).reshape(-1, 1),
                          flat.std(axis=1).reshape(-1, 1),
                          ps.reshape(-1, 1)])

    F_tr = stack_feats(X_tr, pred_tr); F_va = stack_feats(X_va, pred_va); F_te = stack_feats(X_te, pred_te)
    xgb_model = xgb.XGBRegressor(n_estimators=500, max_depth=6, learning_rate=0.03,
                                 subsample=0.8, colsample_bytree=0.8, nthread=8,
                                 early_stopping_rounds=30)
    xgb_model.fit(F_tr, ytr_o, eval_set=[(F_va, yva_o)], verbose=False)
    pred_te_xgb = xgb_model.predict(F_te)

    m_xgb = metrics(pred_te_xgb, yte_o)
    print('\n=== GRU + XGBoost 残差修正 (测试集, 原始尺度 hPa) ===')
    print(f'  RMSE={m_xgb["RMSE"]:.3f}  MAE={m_xgb["MAE"]:.3f}  R2={m_xgb["R2"]:.4f}  MAPE={m_xgb["MAPE"]:.2f}%')

    print('\n=== 对比 ===')
    print(f'  RMSE: GRU {m_gru["RMSE"]:.3f} -> GRU+XGB {m_xgb["RMSE"]:.3f}  (改善 {(m_gru["RMSE"]-m_xgb["RMSE"])/m_gru["RMSE"]*100:+.1f}%)')
    print(f'  MAE : GRU {m_gru["MAE"]:.3f} -> GRU+XGB {m_xgb["MAE"]:.3f}  (改善 {(m_gru["MAE"]-m_xgb["MAE"])/m_gru["MAE"]*100:+.1f}%)')

    pd.DataFrame([{'model': 'GRU', **m_gru}, {'model': 'GRU+XGB', **m_xgb}]).to_csv(
        os.path.join(args.out, 'metrics.csv'), index=False, float_format='%.4f')

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    for ax, (p, t, ti) in zip(axes, [(pred_te, yte_o, 'GRU'), (pred_te_xgb, yte_o, 'GRU+XGB')]):
        ax.scatter(t, p, s=4, alpha=0.2, color='steelblue')
        v0, v1 = t.min(), t.max()
        ax.plot([v0, v1], [v0, v1], 'r--', lw=1)
        m = metrics(p, t)
        ax.set_title(f'{ti}  RMSE={m["RMSE"]:.3f} R2={m["R2"]:.4f}')
        ax.set_xlabel('True PS (hPa)'); ax.set_ylabel('Pred PS (hPa)'); ax.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(args.out, 'scatter_gru_vs_xgb.png'), dpi=200); plt.close()
    print(f'\n完成 -> {args.out}')


if __name__ == '__main__':
    main()
