# -*- coding: utf-8 -*-
"""仅绘图: 用已保存的预测 + 重算真值, 生成 GRU/CNN1D/CNN-BiLSTM-Att/GRU+XGB 对比散点图"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from phase1_model_compare import add_features as _af, create_sequences as _cs
from phase1_gru_xgb_ablation import load_data

DATA = '/share/home/u23114/tj23114/packages/dachuang_pwv/PS/xg_data'
OUT = '/share/home/u23114/tj23114/packages/dachuang_pwv/phase2/result_phase1_compare'
df = load_data(DATA, 200)
df = _af(df, 'PS')
feat_cols = ['doy_sin','doy_cos','hour_sin','hour_cos','month_sin','month_cos',
             'lat_sin','lat_cos','lon_sin','lon_cos','elv_norm','dt_h',
             'PS_lag1','PS_lag2','PS_lag3','PS_lag6','PS_lag12','PS_lag24',
             'PS_diff1','PS_ma6','PS_std6','TS_lag1','TS_lag2','TS_lag3','TS_lag6',
             'TS_diff1','TS_ma6','TS_std6','WPS_lag1','WPS_lag2','WPS_lag3','WPS_lag6',
             'WPS_diff1','WPS_ma6','WPS_std6','PS_TS_ratio','PS_WPS_ratio']
feat_cols = [c for c in feat_cols if c in df.columns]
df = df.dropna(subset=feat_cols)
X, y, meta = _cs(df, feat_cols, 'PS', 12)
times = np.array([m[1] for m in meta], dtype='datetime64[ns]')
order = np.argsort(times)
n = len(order); n_test = int(n*0.2); n_val = int(n*0.15)
te = np.zeros(n, bool); te[order[-n_test:]] = True
yte = y[te]
names = [('GRU','pred_gru.npy'), ('CNN1D','pred_cnn1d.npy'),
         ('CNN-BiLSTM-Att','pred_cnnbilstmattention.npy'), ('GRU+XGB','pred_gruxgb.npy')]
import pandas as pd
mdf = pd.read_csv(os.path.join(OUT, 'metrics.csv')).set_index('model')
fig, axes = plt.subplots(1, 4, figsize=(20, 5))
for ax, (name, fn) in zip(axes, names):
    pred = np.load(os.path.join(OUT, fn))
    ax.scatter(yte, pred, s=4, alpha=0.2, color='steelblue')
    v0, v1 = yte.min(), yte.max()
    ax.plot([v0, v1], [v0, v1], 'r--', lw=1)
    m = mdf.loc[name] if name in mdf.index else {}
    rmse = m.get('RMSE', np.nan); r2 = m.get('R2', np.nan)
    ax.set_title(f'{name}  RMSE={float(rmse):.3f} R2={float(r2):.4f}')
    ax.set_xlabel('True PS (hPa)'); ax.set_ylabel('Pred PS (hPa)'); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT, 'compare_scatter.png'), dpi=200)
print('compare_scatter.png saved')
