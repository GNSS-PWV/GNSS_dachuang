# -*- coding: utf-8 -*-
"""
第一阶段官方指标汇总: 从各参数 year_2017 预测文件计算原始尺度精度
(PS/WPS/Ts/Tm 四个 GRU 模型在 2017 测试年的表现)
"""
import os, sys, glob
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

BASE = '/share/home/u23114/tj23114/packages/dachuang_pwv'
PARAMS = {
    'PS':  f'{BASE}/PS/ps_result_2/gru/predictions/PS/year_2017/*.txt',
    'WPS': f'{BASE}/WPS/wps_result_1/gru/predictions/year_2017/*.txt',
    'Ts':  f'{BASE}/Ts_Tm/ts_result/gru/predictions/TS/year_2017/*.txt',
    'Tm':  f'{BASE}/Ts_Tm/tm_result/gru/predictions/Tm/year_2017/*.txt',
}
UNITS = {'PS': 'hPa', 'WPS': 'hPa', 'Ts': 'K', 'Tm': 'K'}

rows = []
for name, pat in PARAMS.items():
    files = sorted(glob.glob(pat))
    if not files:
        print(f'{name}: 无文件'); continue
    df_list = []
    for fp in files:
        d = pd.read_csv(fp, sep='\s+')
        df_list.append(d)
    df = pd.concat(df_list, ignore_index=True)
    pred = df['Predict'].values; true = df['True'].values
    rmse = float(np.sqrt(mean_squared_error(true, pred)))
    mae = float(mean_absolute_error(true, pred))
    r2 = float(r2_score(true, pred))
    mape = float(np.mean(np.abs((pred - true) / (true + 1e-8))) * 100)
    rows.append({'param': name, 'unit': UNITS[name], 'N': len(df),
                 'RMSE': rmse, 'MAE': mae, 'R2': r2, 'MAPE_pct': mape})
    print(f'{name} ({UNITS[name]}): N={len(df)}  RMSE={rmse:.4f}  MAE={mae:.4f}  R2={r2:.4f}  MAPE={mape:.2f}%')

out = pd.DataFrame(rows)
out.to_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'result_phase1_official_metrics.csv'),
           index=False, float_format='%.6f')
print('\n已保存 result_phase1_official_metrics.csv')
