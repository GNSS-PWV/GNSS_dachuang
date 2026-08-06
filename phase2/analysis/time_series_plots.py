# -*- coding: utf-8 -*-
"""
典型站 PWV 时间序列对比图 (论文/PPT 材料)

对代表性站点绘制一年内 PWV 时间序列: 真值 / ProfileTransformer / GPT3 / 传统(Pi-only)
选站: 一个干站(低PWV), 一个湿站(高PWV), 一个中纬站

输入: test_predictions.csv (官方36站, 含 pwv_true/pwv_pred) + gpt3_predictions.csv
输出: result_ts/ts_<station>.png
"""
import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', required=True, help='test_predictions.csv (含 tm, pwv_true, pwv_pred)')
    ap.add_argument('--gpt3', required=True, help='gpt3_predictions.csv (含 pwv_gpt3, Tm_gpt3)')
    ap.add_argument('--out', default='result_ts')
    ap.add_argument('--year', type=int, default=2014, help='绘制哪一年')
    ap.add_argument('--stations', type=str, default=None, help='指定站点(逗号分隔), 默认自动选干/湿/中纬各一')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    df = pd.read_csv(args.csv)
    g = pd.read_csv(args.gpt3)
    keep = [c for c in ['station_id', 'time', 'pwv_true', 'pwv_pred', 'pwv_gpt3', 'Tm_gpt3'] if c in g.columns]
    g = g[keep]
    # gpt3_predictions.csv 可能有重复列名或额外列, 只保留需要
    df = df.merge(g, on=['station_id', 'time'], how='left', suffixes=('', '_g'))
    df['time'] = pd.to_datetime(df['time'])
    # Pi-only (真值Tm)
    K2P = 22.1; K3 = 3.739e5; RV = 461.495; RHO_W = 1000.0
    df['pwv_pionly'] = (1e8 / (RHO_W * RV * (K3 / df['tm'] + K2P))) * df['zwd']

    if args.stations:
        stations = [s.strip() for s in args.stations.split(',')]
    else:
        per = df.groupby('station_id')['pwv_true'].mean().sort_values()
        stations = [per.index[0], per.index[len(per)//2], per.index[-1]]  # 干/中/湿
    print('选站:', stations)

    for sid in stations:
        sub = df[(df['station_id'] == sid) & (df['time'].dt.year == args.year)].sort_values('time')
        if len(sub) == 0:
            print(f'{sid}: 无 {args.year} 数据, 跳过'); continue
        print(f'{sid}: n={len(sub)}, PWV 均值={sub["pwv_true"].mean():.1f} mm')
        fig, ax = plt.subplots(figsize=(14, 4.5))
        ax.plot(sub['time'], sub['pwv_true'], 'k-', lw=1.2, label='True (radiosonde)')
        ax.plot(sub['time'], sub['pwv_pred'], 'b-', lw=1.0, alpha=0.9, label='ProfileTransformer')
        if 'pwv_gpt3' in sub.columns and sub['pwv_gpt3'].notna().any():
            ax.plot(sub['time'], sub['pwv_gpt3'], color='goldenrod', lw=0.8, alpha=0.8, label='GPT3')
        ax.plot(sub['time'], sub['pwv_pionly'], color='coral', lw=0.8, alpha=0.7, ls='--', label='Pi-only (true Tm)')
        ax.set_xlabel('Time')
        ax.set_ylabel('PWV (mm)')
        ax.set_title(f'{sid}  PWV time series ({args.year})  [mean PWV={sub["pwv_true"].mean():.1f} mm]')
        ax.legend(loc='upper right', fontsize=9, ncol=2)
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(args.out, f'ts_{sid}.png'), dpi=200)
        plt.close()
        print(f'  图已保存: ts_{sid}.png')

    print(f'\n完成 -> {args.out}')


if __name__ == '__main__':
    main()
