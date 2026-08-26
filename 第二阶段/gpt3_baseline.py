# -*- coding: utf-8 -*-
"""
GPT3 基线对比: ProfileTransformer vs GPT3(经验模型) vs Saastamoinen(Pi-only, 真值Tm)

核心对比(项目创新点): 用 GPT3 的加权平均温度 Tm 计算转换系数 Pi, 与真值 ZWD 相乘得到
PWV_gpt3 = Pi(Tm_gpt3) * ZWD, 然后与 ProfileTransformer(完全不用Tm) 对比。

用法:
  python gpt3_baseline.py --csv result/test_predictions.csv --grd ../gpt3_1/gpt3_1.grd --out result_gpt3
"""
import os
import sys
import argparse
import numpy as np
import pandas as pd
import datetime as dt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

K2P = 22.1
K3 = 3.739e5
RV = 461.495
RHO_W = 1000.0


def pi_from_tm(Tm):
    return 1e8 / (RHO_W * RV * (K3 / Tm + K2P))


def compute_metrics(pred, true):
    pred = np.asarray(pred, dtype=np.float64).flatten()
    true = np.asarray(true, dtype=np.float64).flatten()
    rmse = float(np.sqrt(mean_squared_error(true, pred)))
    mae = float(mean_absolute_error(true, pred))
    r2 = float(r2_score(true, pred))
    bias = float(np.mean(pred - true))
    rel = rmse / np.mean(true) * 100 if np.mean(true) > 0 else 0.0
    return {'RMSE': rmse, 'MAE': mae, 'R2': r2, 'Bias': bias, 'Rel_RMSE_pct': rel, 'N': len(true)}


def parse_time(s):
    """解析 'YYYY-MM-DD HH:MM:SS' -> (year, doy, hour)"""
    try:
        t = dt.datetime.strptime(str(s)[:19], '%Y-%m-%d %H:%M:%S')
    except Exception:
        t = pd.to_datetime(s)
    doy = t.timetuple().tm_yday
    return t.year, float(doy), float(t.hour) + float(t.minute) / 60.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', required=True, help='test_predictions.csv (含 station_id/time/lat/lon/elv/zwd/pwv_true/pwv_pred)')
    ap.add_argument('--grd', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'gpt3_1', 'gpt3_1.grd'))
    ap.add_argument('--out', default='result_gpt3')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    # gpt3.py 与 grd 同目录时自动加入路径
    gpt3_dir = os.path.dirname(os.path.abspath(args.grd))
    if gpt3_dir not in sys.path:
        sys.path.insert(0, gpt3_dir)
    from gpt3 import GPT3
    gpt3 = GPT3(args.grd)

    df = pd.read_csv(args.csv)
    print(f'加载预测明细: N={len(df)}', flush=True)

    years, doys, hours = [], [], []
    for s in df['time']:
        y, d, h = parse_time(s)
        years.append(y); doys.append(d); hours.append(h)
    df['year'] = years; df['doy'] = doys; df['hour'] = hours

    print('计算 GPT3 Tm (p/T/Tm/e) ...', flush=True)
    p_g, T_g, Tm_g, e_g = gpt3.compute(df['lat'].values, df['lon'].values,
                                       df['elv'].values, df['year'].values,
                                       df['doy'].values, df['hour'].values)
    df['p_gpt3'] = p_g
    df['T_gpt3'] = T_g
    df['Tm_gpt3'] = Tm_g
    df['e_gpt3'] = e_g

    # GPT3 基线 PWV
    df['pwv_gpt3'] = pi_from_tm(Tm_g) * df['zwd'].values

    # 真值 Tm 基线 (若有 tm 列)
    if 'tm' in df.columns and df['tm'].notna().any():
        df['pwv_pionly'] = pi_from_tm(df['tm'].values) * df['zwd'].values
    else:
        df['pwv_pionly'] = np.nan

    m_trans = compute_metrics(df['pwv_pred'].values, df['pwv_true'].values)
    m_gpt3 = compute_metrics(df['pwv_gpt3'].values, df['pwv_true'].values)
    m_pi = compute_metrics(df['pwv_pionly'].values, df['pwv_true'].values) if df['pwv_pionly'].notna().any() else None

    rows = [
        {'Method': 'Saastamoinen (Pi-only, true Tm)', **m_pi} if m_pi else None,
        {'Method': 'GPT3 (empirical Tm)', **m_gpt3},
        {'Method': 'ProfileTransformer', **m_trans},
    ]
    rows = [r for r in rows if r is not None]
    summary = pd.DataFrame(rows)
    summary.to_csv(os.path.join(args.out, 'gpt3_comparison_metrics.csv'), index=False, float_format='%.6f')

    print('\n' + '=' * 78)
    print(f'{"方法":<28} {"RMSE(mm)":<10} {"MAE(mm)":<10} {"R2":<8} {"Bias(mm)":<10}')
    print('-' * 78)
    for _, r in summary.iterrows():
        print(f'{r["Method"]:<28} {r["RMSE"]:<10.4f} {r["MAE"]:<10.4f} {r["R2"]:<8.4f} {r["Bias"]:<10.4f}')
    print('=' * 78)
    print(f'ProfileTransformer 相比 GPT3 RMSE 改善: {(m_gpt3["RMSE"]-m_trans["RMSE"])/m_gpt3["RMSE"]*100:.1f}%')
    if m_pi:
        print(f'GPT3 相比 Pi-only(真值Tm) RMSE 恶化: {(m_gpt3["RMSE"]-m_pi["RMSE"])/m_pi["RMSE"]*100:.1f}%')

    # Tm 误差分析
    if 'tm' in df.columns and df['tm'].notna().any():
        tm_err = df['Tm_gpt3'].values - df['tm'].values
        print(f'\nGPT3 Tm 误差: RMSE={np.sqrt((tm_err**2).mean()):.2f} K  Bias={tm_err.mean():+.2f} K')
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(df['tm'].values, df['Tm_gpt3'].values, s=3, alpha=0.2, color='coral')
        v0, v1 = df['tm'].min(), df['tm'].max()
        ax.plot([v0, v1], [v0, v1], 'r--', lw=1)
        ax.set_xlabel('True Tm (K)'); ax.set_ylabel('GPT3 Tm (K)')
        ax.set_title(f'GPT3 Tm vs True Tm (RMSE={np.sqrt((tm_err**2).mean()):.2f} K)')
        ax.grid(alpha=0.3)
        plt.tight_layout(); plt.savefig(os.path.join(args.out, 'gpt3_tm_scatter.png'), dpi=200); plt.close()

    # 散点对比图
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for ax, (pred, title, color) in zip(axes, [
        (df['pwv_pionly'].values, 'Saastamoinen (Pi-only, true Tm)', 'coral'),
        (df['pwv_gpt3'].values, 'GPT3 (empirical Tm)', 'goldenrod'),
        (df['pwv_pred'].values, 'ProfileTransformer', 'steelblue'),
    ]):
        if pred is None or np.isnan(pred).all():
            ax.set_visible(False); continue
        ax.scatter(df['pwv_true'].values, pred, s=4, alpha=0.25, c=color)
        vmin = float(min(df['pwv_true'].min(), np.nanmin(pred)))
        vmax = float(max(df['pwv_true'].max(), np.nanmax(pred)))
        ax.plot([vmin, vmax], [vmin, vmax], 'r--', lw=1)
        m = compute_metrics(pred, df['pwv_true'].values)
        ax.set_xlabel('True PWV (mm)'); ax.set_ylabel('Predicted PWV (mm)')
        ax.set_title(f'{title}\nRMSE={m["RMSE"]:.3f} R2={m["R2"]:.4f}')
        ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(args.out, 'gpt3_comparison_scatter.png'), dpi=300); plt.close()

    # 误差分布
    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.linspace(-4, 4, 80)
    ax.hist(df['pwv_gpt3'].values - df['pwv_true'].values, bins=bins, alpha=0.5, label='GPT3', color='goldenrod')
    ax.hist(df['pwv_pred'].values - df['pwv_true'].values, bins=bins, alpha=0.5, label='ProfileTransformer', color='steelblue')
    ax.set_xlabel('PWV Error (mm)'); ax.set_ylabel('Count')
    ax.set_title('Error Distribution: GPT3 vs Transformer')
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(args.out, 'gpt3_error_distribution.png'), dpi=300); plt.close()

    df.to_csv(os.path.join(args.out, 'gpt3_predictions.csv'), index=False, float_format='%.6f')
    print(f'\n结果已保存: {args.out}')


if __name__ == '__main__':
    main()
