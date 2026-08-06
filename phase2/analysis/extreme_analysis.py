# -*- coding: utf-8 -*-
"""
极端场景验证 (使用现有 2014-2019 全球数据中的极端样本, 无需外部数据)

从测试集预测明细中识别"极端天气/水汽条件"样本并统计模型精度:
  1) 特湿:  PWV > 60 mm
  2) 高湿延迟: ZWD > 250 mm
  3) 极值:  按站点 PWV 的前 1% / 5% 分位
  4) 强变化: 同一站相邻两次观测 |dPWV| > 10 mm (急升/急降事件)
  5) 强季变: 相邻两次观测 |dPWV| > 5 mm 且方向一致(连续上升/下降)

可选: 若给出 --grd, 同时计算 GPT3 基线误差, 对比 Transformer 在极端场景的优劣.

输出: result_extreme/extreme_summary.csv, extreme_bar.png, extreme_events.csv
"""
import os
import sys
import argparse
import numpy as np
import pandas as pd
import datetime as dt
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def compute_metrics(pred, true):
    pred = np.asarray(pred, dtype=np.float64).flatten()
    true = np.asarray(true, dtype=np.float64).flatten()
    rmse = float(np.sqrt(mean_squared_error(true, pred)))
    mae = float(mean_absolute_error(true, pred))
    r2 = float(r2_score(true, pred))
    bias = float(np.mean(pred - true))
    rel = rmse / np.mean(true) * 100 if np.mean(true) > 0 else 0.0
    return {'RMSE': rmse, 'MAE': mae, 'R2': r2, 'Bias': bias, 'Rel_RMSE_pct': rel, 'N': len(true)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', required=True)
    ap.add_argument('--out', default='result_extreme')
    ap.add_argument('--grd', default=None, help='给出则同时计算 GPT3 基线')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    df = pd.read_csv(args.csv)
    df = df.sort_values(['station_id', 'time']).reset_index(drop=True)
    print(f'加载: N={len(df)}', flush=True)

    if args.grd and os.path.exists(args.grd):
        gpt3_dir = os.path.dirname(os.path.abspath(args.grd))
        if gpt3_dir not in sys.path:
            sys.path.insert(0, gpt3_dir)
        from gpt3 import GPT3
        gpt3 = GPT3(args.grd)
        years, doys, hours = [], [], []
        for s in df['time']:
            t = dt.datetime.strptime(str(s)[:19], '%Y-%m-%d %H:%M:%S')
            years.append(t.year); doys.append(t.timetuple().tm_yday)
            hours.append(t.hour + t.minute / 60.0)
        _, _, Tm_g, _ = gpt3.compute(df['lat'].values, df['lon'].values, df['elv'].values,
                                     np.array(years), np.array(doys), np.array(hours))
        K2P = 22.1; K3 = 3.739e5; RV = 461.495; RHO_W = 1000.0
        pi = 1e8 / (RHO_W * RV * (K3 / Tm_g + K2P))
        df['pwv_gpt3'] = pi * df['zwd'].values
        print('GPT3 基线已计算', flush=True)
    else:
        df['pwv_gpt3'] = np.nan

    # 相邻观测变化量 (同站)
    df['dPWV'] = df.groupby('station_id')['pwv_true'].diff()
    df['dPWV_abs'] = df['dPWV'].abs()

    subsets = {
        'All': df,
        'Wet PWV>60mm': df[df['pwv_true'] > 60],
        'ZWD>250mm': df[df['zwd'] > 250],
        'Rapid rise dPWV>+10mm': df[df['dPWV'] > 10],
        'Rapid drop dPWV<-10mm': df[df['dPWV'] < -10],
        'Rapid change |dPWV|>10mm': df[df['dPWV_abs'] > 10],
        'Consecutive rise': df[(df['dPWV'] > 5)],
        'Station PWV top1%': pd.concat([g[g['pwv_true'] >= g['pwv_true'].quantile(0.99)]
                                 for _, g in df.groupby('station_id')]),
        'Station PWV top5%': pd.concat([g[g['pwv_true'] >= g['pwv_true'].quantile(0.95)]
                                 for _, g in df.groupby('station_id')]),
    }

    rows = []
    for name, sub in subsets.items():
        m = compute_metrics(sub['pwv_pred'].values, sub['pwv_true'].values)
        m['subset'] = name
        m['pwv_mean'] = float(sub['pwv_true'].mean())
        if sub['pwv_gpt3'].notna().any():
            mg = compute_metrics(sub['pwv_gpt3'].values, sub['pwv_true'].values)
            m['gpt3_RMSE'] = mg['RMSE']
            m['gpt3_Bias'] = mg['Bias']
            m['impr_vs_gpt3_pct'] = (mg['RMSE'] - m['RMSE']) / mg['RMSE'] * 100
        else:
            m['gpt3_RMSE'] = np.nan; m['gpt3_Bias'] = np.nan; m['impr_vs_gpt3_pct'] = np.nan
        rows.append(m)
    summary = pd.DataFrame(rows)
    summary.to_csv(os.path.join(args.out, 'extreme_summary.csv'), index=False, float_format='%.6f')
    print(summary.to_string(index=False))

    # 极端事件明细
    ev = df[df['dPWV_abs'] > 10].copy()
    ev['abs_err'] = (ev['pwv_pred'] - ev['pwv_true']).abs()
    ev = ev.sort_values('abs_err', ascending=False)
    ev_out = ev[['station_id', 'time', 'lat', 'lon', 'zwd', 'pwv_true', 'pwv_pred', 'pwv_error', 'dPWV']].head(100)
    ev_out.to_csv(os.path.join(args.out, 'extreme_events.csv'), index=False, float_format='%.4f')
    print(f'\n强变化事件数: {len(ev)}, 已保存前100条: extreme_events.csv')

    # 柱状图: 各极端子集 RMSE + 相对 GPT3 提升
    plot = summary[summary['subset'] != 'All']
    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(plot))
    ax.bar(x, plot['RMSE'].values, color='steelblue', label='Transformer RMSE')
    if plot['gpt3_RMSE'].notna().any():
        ax.plot(x, plot['gpt3_RMSE'].values, 'ro-', label='GPT3 RMSE')
    ax.set_xticks(x); ax.set_xticklabels(plot['subset'].values, rotation=25, ha='right', fontsize=8)
    ax.set_ylabel('RMSE (mm)')
    ax.set_title('Extreme-condition PWV RMSE (Transformer vs GPT3)')
    ax.legend(); ax.grid(alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(args.out, 'extreme_bar.png'), dpi=200)
    plt.close()

    print(f'\n全部完成 -> {args.out}')


if __name__ == '__main__':
    main()
