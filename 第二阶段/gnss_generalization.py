# -*- coding: utf-8 -*-
"""
GNSS ZPD -> PWV 泛化验证管线 (部署场景, 全本地可跑)
====================================================
输入: IGS SINEX TROP ZPD (2019, 5min ZTD) + 月尺度全球 Pi 格网 + GPT3
流程: 解析ZTD -> 站点坐标 -> ZHD(GPT3气压, Saastamoinen干延迟) -> ZWD
      -> Pi(月尺度格网查表) -> PWV = Pi * ZWD
"""
import os, sys, glob, gzip
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

BASE = r'D:\gnss水汽反演'
ZPD_DIR = os.path.join(BASE, '2019')
GRID_CSV = os.path.join(BASE, '第二阶段', 'result_grid_monthly', 'pi_grid_all.csv')
GRD = os.path.join(BASE, 'gpt3_1', 'gpt3_1.grd')
COORD_FILE = os.path.join(BASE, 'IGSwhu_formatted(1).txt')
OUT = os.path.dirname(os.path.abspath(__file__))

# 中国代表站 (武汉/香港/上海佘山/厦门/北京/拉萨/乌鲁木齐/长春)
TARGETS = ['wuh2', 'hksl', 'hkws', 'shao', 'kmnm', 'bjfs', 'lhaz', 'urum', 'chan']


def load_coords():
    """从 IGSwhu_formatted(1).txt 读取站点坐标 {4位站码: (lat, lon, h)}."""
    coords = {}
    with open(COORD_FILE, encoding='utf-8') as fh:
        for ln in fh:
            ln = ln.strip()
            if not ln or ',' not in ln:
                continue
            p = [x.strip() for x in ln.split(',')]
            if len(p) >= 4:
                try:
                    coords[p[0][:4].lower()] = (float(p[1]), float(p[2]), float(p[3]))
                except Exception:
                    pass
    return coords


def parse_site(f):
    st = os.path.basename(f)[:4].upper()
    try:
        with gzip.open(f, 'rt', encoding='utf-8', errors='replace') as fh:
            for ln in fh:
                if ln.startswith(' ' + st):
                    p = ln.split()
                    if len(p) >= 14:
                        lo_d, lo_m, lo_s = float(p[-7]), float(p[-6]), float(p[-5])
                        la_d, la_m, la_s = float(p[-4]), float(p[-3]), float(p[-2])
                        lon = lo_d + lo_m/60 + lo_s/3600
                        if lon > 180: lon -= 360
                        lat = la_d + la_m/60 + la_s/3600
                        return lat, lon, float(p[-1])
                    return None
    except Exception:
        return None
    return None


def parse_zpd(f):
    epochs, ztd = [], []
    with gzip.open(f, 'rt', encoding='utf-8', errors='replace') as fh:
        in_sol = False
        for ln in fh:
            if '+TROP/SOLUTION' in ln:
                in_sol = True; continue
            if '-TROP/SOLUTION' in ln:
                break
            if in_sol and ln.startswith(' '):
                p = ln.split()
                if len(p) >= 4 and ':' in p[1]:
                    try:
                        epochs.append(int(p[1].split(':')[2]))
                        ztd.append(float(p[2]))
                    except Exception:
                        pass
    return np.array(epochs), np.array(ztd)


def main():
    grid = pd.read_csv(GRID_CSV)
    grid['month'] = grid['season'].str.extract(r'(\d+)').astype(int)
    print(f'Pi 格网: {len(grid)} 行, 月份 {grid["month"].min()}-{grid["month"].max()}', flush=True)

    gpt3_dir = os.path.dirname(os.path.abspath(GRD))
    if gpt3_dir not in sys.path:
        sys.path.insert(0, gpt3_dir)
    from gpt3 import GPT3
    gpt3 = GPT3(GRD)

    days = sorted([d for d in os.listdir(ZPD_DIR) if os.path.isdir(os.path.join(ZPD_DIR, d))])
    print(f'天数: {len(days)} ({days[0]}-{days[-1]})', flush=True)

    station_data = {t: {'times': [], 'ztd': []} for t in TARGETS}
    coord = load_coords()
    for day in days:
        d = int(day)
        for f in glob.glob(os.path.join(ZPD_DIR, day, '*.gz')):
            st = os.path.basename(f)[:4].lower()
            if st not in TARGETS:
                continue
            if st not in coord:
                c = parse_site(f)
                if c:
                    coord[st] = c
            epochs, ztd = parse_zpd(f)
            if len(ztd) == 0:
                continue
            t0 = pd.Timestamp(year=2019, month=1, day=1) + pd.Timedelta(days=d-1)
            station_data[st]['times'].extend(t0 + pd.to_timedelta(epochs, unit='s'))
            station_data[st]['ztd'].extend(ztd)
    print(f'站点坐标: {coord}', flush=True)

    results = {}
    for st in TARGETS:
        if len(station_data[st]['ztd']) == 0 or st not in coord:
            print(f'{st}: 无数据或坐标'); continue
        df = pd.DataFrame({'time': station_data[st]['times'], 'ztd_mm': station_data[st]['ztd']})
        df = df.sort_values('time').drop_duplicates('time').reset_index(drop=True)
        lat, lon, h = coord[st]
        df['doy'] = df['time'].dt.dayofyear
        df['hour'] = df['time'].dt.hour + df['time'].dt.minute/60
        df['month'] = df['time'].dt.month
        # GPT3 地表气压 (向量化, 全部样本)
        p, T, Tm, e = gpt3.compute(np.full(len(df), lat), np.full(len(df), lon),
                                   np.full(len(df), h), np.full(len(df), 2019),
                                   df['doy'].values, df['hour'].values)
        df['ps'] = np.asarray(p)
        # ZHD (Saastamoinen 干延迟)
        f = 1.0 - 0.00266*np.cos(2*np.deg2rad(lat)) - 0.00000028*h
        df['zhd_mm'] = 0.002277 * df['ps'] / f * 1000.0
        df['zwd_mm'] = df['ztd_mm'] - df['zhd_mm']
        # Pi 查月尺度格网 (按月份查最近格点)
        pi_series = np.zeros(len(df))
        for i, r in df.iterrows():
            gm = grid[grid['month'] == int(r['month'])]
            d2 = (gm['lat'] - lat)**2 + (((gm['lon'] - lon + 180) % 360 - 180))**2
            pi_series[i] = gm.loc[d2.idxmin(), 'pi']
        df['pi'] = pi_series
        df['pwv_mm'] = df['pi'] * df['zwd_mm']
        df = df.dropna(subset=['pwv_mm'])
        results[st] = df
        print(f'{st}: N={len(df)}  ZTD[{df.ztd_mm.min():.0f},{df.ztd_mm.max():.0f}]  '
              f'PWV mean={df.pwv_mm.mean():.2f} max={df.pwv_mm.max():.2f}mm', flush=True)
        df[['time', 'ztd_mm', 'zhd_mm', 'zwd_mm', 'pi', 'pwv_mm']].to_csv(
            os.path.join(OUT, f'gnss_pwv_{st}.csv'), index=False, float_format='%.3f')

    fig, axes = plt.subplots(len(results), 1, figsize=(15, 3.5*len(results)), sharex=True)
    if len(results) == 1:
        axes = [axes]
    for ax, (st, df) in zip(axes, results.items()):
        dly = df.set_index('time')['pwv_mm'].resample('1h').mean()
        ax.plot(dly.index, dly.values, lw=0.8, color='steelblue')
        ax.set_ylabel('PWV (mm)')
        ax.set_title(f'{st} (lat={coord[st][0]:.2f}, lon={coord[st][1]:.2f})  GNSS PWV via gridded Pi')
        ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, 'gnss_pwv_timeseries.png'), dpi=150)
    print('\n完成: gnss_pwv_*.csv + gnss_pwv_timeseries.png')


if __name__ == '__main__':
    main()
