# -*- coding: utf-8 -*-
import os, sys
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from phase1_gru_xgb_ablation import load_data, add_features, create_sequences

DATA = '/share/home/u23114/tj23114/packages/dachuang_pwv/PS/xg_data'
df = load_data(DATA, 80)
print('raw rows:', len(df), 'stations:', df['station_id'].nunique())
cnt = df.groupby('station_id').size()
print('per-station obs: min=%d max=%d mean=%.0f' % (cnt.min(), cnt.max(), cnt.mean()))
print('time range:', df['TIME'].min(), '->', df['TIME'].max())
dt = df.groupby('station_id')['TIME'].diff().dt.total_seconds()/3600
print('sampling interval (h) value counts:')
print(dt.value_counts().head(8))

df = add_features(df, 'PS')
print('after features:', len(df))
c1 = df['PS'].corr(df['PS_lag1'])
c2 = df['PS'].corr(df['PS_lag2'])
print('corr(PS,PS_lag1)=%.4f  corr(PS,PS_lag2)=%.4f' % (c1, c2))

feat_cols = ['doy_sin','doy_cos','hour_sin','hour_cos','month_sin','month_cos',
             'lat_sin','lat_cos','lon_sin','lon_cos','elv_norm',
             'PS_lag1','PS_lag2','PS_lag3','PS_lag6','PS_lag12','PS_lag24',
             'PS_diff1','PS_ma6','PS_std6','dt_h']
feat_cols = [c for c in feat_cols if c in df.columns]
df = df.dropna(subset=feat_cols)
X, y, meta = create_sequences(df, feat_cols, 'PS', 12)
print('X:', X.shape)
times = np.array([m[1] for m in meta], dtype='datetime64[ns]')
order = np.argsort(times)
n = len(order); n_test = int(n*0.2); n_val = int(n*0.15)
te_idx = order[-n_test:]
last_lag1 = X[te_idx, -1, feat_cols.index('PS_lag1')]
y_te = y[te_idx]
rmse = float(np.sqrt(((last_lag1 - y_te)**2).mean()))
r2 = 1 - ((last_lag1-y_te)**2).sum() / ((y_te-y_te.mean())**2).sum()
print('[persistence pred=PS_lag1] RMSE=%.3f hPa  R2=%.4f' % (rmse, r2))
lag2 = X[te_idx, -1, feat_cols.index('PS_lag2')]
pred_ex = last_lag1 + (last_lag1 - lag2)
rmse2 = float(np.sqrt(((pred_ex - y_te)**2).mean()))
r2b = 1 - ((pred_ex-y_te)**2).sum() / ((y_te-y_te.mean())**2).sum()
print('[linear extrapolation] RMSE=%.3f hPa  R2=%.4f' % (rmse2, r2b))
print('test period:', times[te_idx].min(), '->', times[te_idx].max())
print('train period:', times[order[:-(n_test+n_val)]].min(), '->', times[order[:-(n_test+n_val)]].max())
