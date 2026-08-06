# -*- coding: utf-8 -*-
"""
第一阶段模型横向对比 (干净复现): GRU vs CNN1D vs CNN-BiLSTM-Attention vs GRU+XGBoost
=====================================================================================
目标: 回答"加 CNN 后为何效果反而下降" + "GRU+XGB 提升多少"
设定: 与第一阶段官方一致
  - 数据: 探空文件只取地表层(最小ELV), 构成(站点, 时刻)时间序列
  - 切分: 按时间顺序切分 (同站预测未来)  train 65% / val 15% / test 20%
  - 特征: 时间周期 + 站点 + PS/TS/WPS 滞后/差分/滑动 + 交互项
模型:
  - GRU                 (纯 GRU 基线)
  - CNN1D               (干净的一维卷积 + 全局池化)
  - CNN-BiLSTM-Attention(还原第一阶段尝试的复杂混合结构)
  - GRU+XGB             (官方 gru_ps.py 的 stacking: 展平序列+统计+GRU预测 -> XGB直接预测PS)
用法:
  python phase1_model_compare.py --data_dir <xg_data> --max_files 200 --epochs 60
"""
import os
import sys
import argparse
import numpy as np
import pandas as pd
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
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from phase1_gru_xgb_ablation import load_data  # 已修复: 只取地表层


# ---------------- 模型 ----------------
class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size=96, num_layers=2, output_size=1, time_steps=12, dropout=0.3):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True,
                          dropout=dropout if num_layers > 1 else 0)
        self.ln = nn.LayerNorm(hidden_size)
        self.fc = nn.Sequential(nn.Linear(hidden_size, 64), nn.ReLU(), nn.Dropout(dropout),
                                nn.Linear(64, output_size))

    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc(self.ln(out[:, -1, :]))


class CNN1D(nn.Module):
    """干净的一维 CNN: 2层卷积 + 全局平均池化 + FC (参数量与GRU相当)"""
    def __init__(self, input_size, hidden_size=96, time_steps=12, output_size=1, dropout=0.3):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(input_size, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.fc = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Dropout(dropout),
                                nn.Linear(64, output_size))

    def forward(self, x):
        x = x.transpose(1, 2)          # (B, F, T)
        x = self.cnn(x)                # (B, 128, T)
        x = x.mean(dim=2)              # 全局平均池化 (B, 128)
        return self.fc(x)


class MultiHeadAttention(nn.Module):
    """与第一阶段 gru_cnn.py 类似的简易多头注意力"""
    def __init__(self, hidden_dim, num_heads=4, dropout=0.3):
        super().__init__()
        assert hidden_dim % num_heads == 0
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.q = nn.Linear(hidden_dim, hidden_dim)
        self.k = nn.Linear(hidden_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, hidden_dim)
        self.drop = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5

    def forward(self, x):
        B, T, D = x.shape
        q = self.q(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = torch.softmax(attn, dim=-1)
        attn = self.drop(attn)
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, T, D)
        return self.out(out)


class CNNBiLSTMAttention(nn.Module):
    """还原第一阶段尝试的 CNN-BiLSTM-Attention 混合结构"""
    def __init__(self, input_size, hidden_size=128, num_layers=3, output_size=1,
                 time_steps=12, dropout=0.25, num_heads=4):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(input_size, 64, kernel_size=3, padding=1),
            nn.GroupNorm(8, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.GroupNorm(8, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.GroupNorm(8, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.MaxPool1d(kernel_size=2),
        )
        self.lstm = nn.LSTM(256, hidden_size, num_layers, batch_first=True,
                            bidirectional=True, dropout=dropout if num_layers > 1 else 0)
        self.attention = MultiHeadAttention(hidden_size * 2, num_heads=num_heads, dropout=0.3)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2 * 2, 256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 32), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(32, output_size),
        )

    def forward(self, x):
        x_cnn = x.transpose(1, 2)
        x_cnn = self.cnn(x_cnn)
        x_lstm = x_cnn.transpose(1, 2)
        lstm_out, _ = self.lstm(x_lstm)
        attn_out = self.attention(lstm_out)
        avg_pool = torch.mean(attn_out, dim=1)
        max_pool = torch.max(attn_out, dim=1)[0]
        pooled = torch.cat([avg_pool, max_pool], dim=1)
        return self.fc(pooled)


# ---------------- 特征 ----------------
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
    d['dt_h'] = d.groupby('station_id')['TIME'].diff().dt.total_seconds() / 3600
    g = d.groupby('station_id')
    for var, lags in [('PS', [1, 2, 3, 6, 12, 24]), ('TS', [1, 2, 3, 6]), ('WPS', [1, 2, 3, 6])]:
        for lag in lags:
            d[f'{var}_lag{lag}'] = g[var].shift(lag)
        d[f'{var}_diff1'] = d[var] - d[f'{var}_lag1']
        d[f'{var}_ma6'] = g[var].transform(lambda s: s.rolling(6, min_periods=1).mean())
        d[f'{var}_std6'] = g[var].transform(lambda s: s.rolling(6, min_periods=1).std())
    d['PS_TS_ratio'] = d['PS'] / (d['TS'] + 1e-8)
    d['PS_WPS_ratio'] = d['PS'] / (d['WPS'] + 1e-8)
    return d


def create_sequences(df, feat_cols, target, time_steps):
    X, y, meta = [], [], []
    for _, g in df.groupby('station_id'):
        vals = g[feat_cols].values
        ys = g[target].values
        for i in range(time_steps, len(g)):
            X.append(vals[i - time_steps:i])
            y.append(ys[i])
            meta.append((g['station_id'].iloc[i], g['TIME'].iloc[i]))
    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.float32), meta


def metrics(pred, true):
    pred = np.asarray(pred); true = np.asarray(true)
    rmse = float(np.sqrt(mean_squared_error(true, pred)))
    mae = float(mean_absolute_error(true, pred))
    r2 = float(r2_score(true, pred))
    mape = float(np.mean(np.abs((pred - true) / (true + 1e-8))) * 100)
    return {'RMSE': rmse, 'MAE': mae, 'R2': r2, 'MAPE': mape, 'N': len(true)}


def train_model(model, tr_loader, va_loader, epochs, device, lr=1e-3):
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-5)
    crit = nn.SmoothL1Loss(beta=1.0)
    best_val = float('inf'); best_state = None
    for ep in range(epochs):
        model.train()
        for xb, yb in tr_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward(); opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            vp = np.concatenate([model(xb.to(device)).cpu().numpy() for xb, _ in va_loader]).flatten()
        yva = np.concatenate([yb.numpy() for _, yb in va_loader]).flatten()
        vr = float(np.sqrt(mean_squared_error(yva, vp)))
        if vr < best_val:
            best_val = vr
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model, best_val


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', required=True)
    ap.add_argument('--out', default='result_phase1_compare')
    ap.add_argument('--max_files', type=int, default=200)
    ap.add_argument('--time_steps', type=int, default=12)
    ap.add_argument('--epochs', type=int, default=60)
    ap.add_argument('--batch_size', type=int, default=256)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'设备: {device}', flush=True)

    df = load_data(args.data_dir, args.max_files)
    print(f'地表层数据: {len(df)} 行, {df["station_id"].nunique()} 站', flush=True)
    df = add_features(df, 'PS')
    feat_cols = ['doy_sin','doy_cos','hour_sin','hour_cos','month_sin','month_cos',
                 'lat_sin','lat_cos','lon_sin','lon_cos','elv_norm','dt_h',
                 'PS_lag1','PS_lag2','PS_lag3','PS_lag6','PS_lag12','PS_lag24',
                 'PS_diff1','PS_ma6','PS_std6',
                 'TS_lag1','TS_lag2','TS_lag3','TS_lag6','TS_diff1','TS_ma6','TS_std6',
                 'WPS_lag1','WPS_lag2','WPS_lag3','WPS_lag6','WPS_diff1','WPS_ma6','WPS_std6',
                 'PS_TS_ratio','PS_WPS_ratio']
    feat_cols = [c for c in feat_cols if c in df.columns]
    n_before = len(df)
    df = df.dropna(subset=feat_cols)
    print(f'特征行清理: {n_before} -> {len(df)}', flush=True)
    X, y, meta = create_sequences(df, feat_cols, 'PS', args.time_steps)
    print(f'序列: X={X.shape}, 特征数={len(feat_cols)}', flush=True)

    times = np.array([m[1] for m in meta], dtype='datetime64[ns]')
    order = np.argsort(times)
    n = len(order); n_test = int(n * 0.2); n_val = int(n * 0.15)
    tr = np.zeros(n, bool); va = np.zeros(n, bool); te = np.zeros(n, bool)
    tr[order[:-(n_test + n_val)]] = True
    va[order[-(n_test + n_val):-n_test]] = True
    te[order[-n_test:]] = True
    print(f'切分(时间): 训练{tr.sum()} 验证{va.sum()} 测试{te.sum()}', flush=True)

    x_scaler = RobustScaler(); x_scaler.fit(X[tr].reshape(-1, X.shape[2]))
    X_tr = x_scaler.transform(X[tr].reshape(-1, X.shape[2])).reshape(-1, args.time_steps, X.shape[2])
    X_va = x_scaler.transform(X[va].reshape(-1, X.shape[2])).reshape(-1, args.time_steps, X.shape[2])
    X_te = x_scaler.transform(X[te].reshape(-1, X.shape[2])).reshape(-1, args.time_steps, X.shape[2])
    y_scaler = RobustScaler(); y_scaler.fit(y[tr].reshape(-1, 1))
    y_tr = y_scaler.transform(y[tr].reshape(-1, 1)).flatten()
    y_va = y_scaler.transform(y[va].reshape(-1, 1)).flatten()
    yte_o = y[te]

    tr_ds = TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr).unsqueeze(1))
    va_ds = TensorDataset(torch.from_numpy(X_va), torch.from_numpy(y_va).unsqueeze(1))
    tr_loader = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True)
    va_loader = DataLoader(va_ds, batch_size=args.batch_size)

    def predict(model, Xa):
        model.eval()
        with torch.no_grad():
            p = model(torch.from_numpy(Xa).to(device)).cpu().numpy().flatten()
        return y_scaler.inverse_transform(p.reshape(-1, 1)).flatten()

    FNAME_MAP = {'GRU': 'pred_gru.npy', 'CNN1D': 'pred_cnn1d.npy',
                'CNN-BiLSTM-Attention': 'pred_cnnbilstmattention.npy', 'GRU+XGB': 'pred_gruxgb.npy'}
    results = {}
    models = {
        'GRU': GRUModel(X.shape[2], hidden_size=96, num_layers=2, time_steps=args.time_steps),
        'CNN1D': CNN1D(X.shape[2], hidden_size=96, time_steps=args.time_steps),
        'CNN-BiLSTM-Attention': CNNBiLSTMAttention(X.shape[2], hidden_size=128, num_layers=3,
                                                   time_steps=args.time_steps),
    }
    for name, model in models.items():
        print(f'\n===== 训练 {name} ({sum(p.numel() for p in model.parameters()):,} 参数) =====', flush=True)
        model.to(device)
        train_model(model, tr_loader, va_loader, args.epochs, device)
        pred_te = predict(model, X_te)
        m = metrics(pred_te, yte_o)
        results[name] = m
        print(f'{name}: RMSE={m["RMSE"]:.3f}  MAE={m["MAE"]:.3f}  R2={m["R2"]:.4f}  MAPE={m["MAPE"]:.2f}%', flush=True)
        np.save(os.path.join(args.out, FNAME_MAP[name]), pred_te)

    # GRU+XGB (官方 stacking 公式)
    print('\n===== GRU + XGBoost (官方 stacking) =====', flush=True)
    import xgboost as xgb
    gmodel = GRUModel(X.shape[2], hidden_size=96, num_layers=2, time_steps=args.time_steps).to(device)
    train_model(gmodel, tr_loader, va_loader, args.epochs, device)
    p_tr = predict(gmodel, X_tr); p_va = predict(gmodel, X_va); p_te = predict(gmodel, X_te)
    ytr_o = y[tr]; yva_o = y[va]

    def stack_feats(Xs, ps):
        flat = Xs.reshape(Xs.shape[0], -1)
        return np.hstack([flat, flat.mean(axis=1).reshape(-1, 1),
                          flat.std(axis=1).reshape(-1, 1), ps.reshape(-1, 1)])

    F_tr = stack_feats(X_tr, p_tr); F_va = stack_feats(X_va, p_va); F_te = stack_feats(X_te, p_te)
    xm = xgb.XGBRegressor(n_estimators=800, max_depth=6, learning_rate=0.02,
                          subsample=0.8, colsample_bytree=0.8, nthread=8, early_stopping_rounds=30)
    xm.fit(F_tr, ytr_o, eval_set=[(F_va, yva_o)], verbose=False)
    pred_xgb = xm.predict(F_te)
    m_xgb = metrics(pred_xgb, yte_o)
    results['GRU+XGB'] = m_xgb
    print(f'GRU+XGB: RMSE={m_xgb["RMSE"]:.3f}  MAE={m_xgb["MAE"]:.3f}  R2={m_xgb["R2"]:.4f}  MAPE={m_xgb["MAPE"]:.2f}%', flush=True)
    np.save(os.path.join(args.out, 'pred_gruxgb.npy'), pred_xgb)
    np.save(os.path.join(args.out, 'y_true_test.npy'), yte_o)

    # 汇总表
    summary = pd.DataFrame([{'model': k, **v} for k, v in results.items()])
    summary.to_csv(os.path.join(args.out, 'metrics.csv'), index=False, float_format='%.4f')
    print('\n===== 汇总 =====')
    print(summary.to_string(index=False))
    base = results['GRU']['RMSE']
    for name, m in results.items():
        if name != 'GRU':
            print(f'  vs GRU: {name} RMSE 相对变化 {(m["RMSE"]-base)/base*100:+.1f}%')

    # 散点图
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    for ax, (name, m) in zip(axes, results.items()):
        pred = np.load(os.path.join(args.out, FNAME_MAP[name]))
        ax.scatter(yte_o, pred, s=4, alpha=0.2, color='steelblue')
        v0, v1 = yte_o.min(), yte_o.max()
        ax.plot([v0, v1], [v0, v1], 'r--', lw=1)
        ax.set_title(f'{name}  RMSE={m["RMSE"]:.3f} R2={m["R2"]:.4f}')
        ax.set_xlabel('True PS (hPa)'); ax.set_ylabel('Pred PS (hPa)'); ax.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(args.out, 'compare_scatter.png'), dpi=200); plt.close()
    print(f'\n完成 -> {args.out}')


if __name__ == '__main__':
    main()
