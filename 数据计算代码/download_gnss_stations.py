"""
按日期区间下载全部 GNSS 数据源, 并预处理为**按站点**的 ZTD/PWV 时序文件。

与 download_gnss_by_fy3g.py 的区别:
    - 那个脚本: 按 FY-3G 过境日期下载, 输出每天一个 GNSS_PWV_YYYYMMDD.nc (训练标签)
    - 这个脚本: 按 --start/--end 日期区间下载, 输出每站一个 CSV (站点级时序库)
    两者互不干扰, 本脚本不修改也不读取那个脚本的输出。

数据源 (全部 5 个, 取自 GNSSDownloader.DATA_SOURCES):
    igs_whu      全球骨干, ZTD, 免认证, 逐站逐日下载 (最慢)
    euref        欧洲,     ZTD, 免认证, 整网文件
    australia    澳洲+亚太, ZTD, 免认证, 整网文件
    ucar_cosmic  美国CONUS, PWV直接产品 (也含ZTD), 免认证
    suominet     美国高密度, ZTD, 需 EarthScope OAuth2 认证 (--sources 显式指定才启用)

输出: <output_dir>/<站点名>_<起始时间>_<结束时间>.csv
    datetime,ztd_mm,pwv_mm
    202311010000,2456.312,18.442
    - datetime = 年月日时分 (默认 %Y%m%d%H%M)
    - 该站有任何有效 PWV 才输出 pwv_mm 列 ("有pwv的再放一列")
    - ZTD 或 PWV 任一有效即保留, 不因缺 PWV 丢掉 ZTD

用法:
    # 全部免认证源, 2023年11月
    python download_gnss_stations.py --start 20231101 --end 20231130

    # 含 SuomiNet (需先配好 EarthScope 认证)
    python download_gnss_stations.py --start 20231101 --end 20231130 \\
        --sources igs_whu,euref,australia,ucar_cosmic,suominet

    # 只要 ZTD, 不算 PWV (最快)
    python download_gnss_stations.py --start 20231101 --end 20231130 --no_pwv
"""

import os
import re
import sys
import json
import argparse
import warnings
from pathlib import Path
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import xarray as xr
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    from src.data.download.gnss_downloader import GNSSDownloader
except ModuleNotFoundError as exc:
    # 兼容数据计算代码目录内提供的独立下载器；其他依赖缺失仍应原样抛出。
    if exc.name not in {"src", "src.data", "src.data.download", "src.data.download.gnss_downloader"}:
        raise
    from gnss_downloader import GNSSDownloader

# ---- 物理常数 (与 download_gnss_by_fy3g.py 一致, 此处内联以解除耦合) ----
G_ACC = 9.80665      # 重力加速度 m/s²
M_AIR = 0.0289644    # 干空气摩尔质量 kg/mol
R_GAS = 8.31447      # 气体常数 J/(mol·K)
L_RATE = 0.0065      # 温度递减率 K/m

TIME_FMT = '%Y%m%d%H%M'          # 输出 datetime 格式: 年月日时分
ALL_SOURCES = ['igs_whu', 'euref', 'australia', 'ucar_cosmic', 'suominet']
AUTH_SOURCES = {'suominet'}      # 需认证, 默认不启用
DEFAULT_SOURCES = [s for s in ALL_SOURCES if s not in AUTH_SOURCES]

_EGM96_PATH = (Path(__file__).resolve().parent / 'download_lables' /
               'data' / 'geoids' / 'egm96-5.pgm')
_egm96 = None
_egm96_failed = False


def get_egm96():
    """懒加载 EGM96; 缺文件不抛错, 返回 None 由调用方降级"""
    global _egm96, _egm96_failed
    if _egm96 is not None or _egm96_failed:
        return _egm96
    try:
        from pygeodesy.geoids import GeoidKarney
        _egm96 = GeoidKarney(str(_EGM96_PATH))
        print('  已加载 EGM96 大地水准面模型')
    except Exception as e:
        _egm96_failed = True
        print(f'  [WARN] EGM96 加载失败 ({e}); 高程按椭球高处理, '
              f'PWV 会有小偏差, ZTD 不受影响')
    return _egm96


def to_orthometric(lats, lons, h_ellip):
    """椭球高 → 正高 (与 ERA5 高程系统一致); EGM96 不可用时原值返回"""
    egm = get_egm96()
    if egm is None:
        return np.asarray(h_ellip, dtype=float), None
    from pygeodesy.points import LatLon_
    n = np.full(len(lats), np.nan)
    for i, (la, lo) in enumerate(zip(lats, lons)):
        if np.isfinite(la) and np.isfinite(lo):
            try:
                n[i] = egm(LatLon_(float(la), float(lo)))
            except Exception:
                pass
    return np.asarray(h_ellip, dtype=float) - n, n


def safe_station_name(sid):
    """站点ID → 文件名安全字符串"""
    s = re.sub(r'[^0-9A-Za-z_.-]', '_', str(sid).strip())
    return s.upper() if s else 'UNKNOWN'


def daterange(start, end):
    """[start, end] 逐日"""
    d = start
    while d <= end:
        yield d
        d += timedelta(days=1)

def _attach_coords(df, dl, source):
    """用本地站点列表补 lat/lon/height (多数源的解析结果不含坐标)"""
    if df is None or df.empty:
        return df
    need = [c for c in ('lat', 'lon', 'height') if c not in df.columns
            or df[c].isna().all()]
    if not need:
        return df
    st = dl.load_station_list_from_file(source)
    if st is None or st.empty:
        return df
    st = st.copy()
    st['_sid'] = st['station_id'].astype(str).str[:4].str.upper()
    cmap = st.drop_duplicates('_sid').set_index('_sid')
    key = df['station_id'].astype(str).str[:4].str.upper()
    for c in need:
        if c in cmap.columns:
            df[c] = key.map(cmap[c])
    before = len(df)
    df = df.dropna(subset=[c for c in ('lat', 'lon') if c in df.columns])
    if before > len(df):
        print(f'    移除 {before - len(df)} 条无坐标记录')
    return df


def fetch_euref(dl, date, lat_range=None, lon_range=None):
    df = dl.download_and_parse_euref(date=date, lat_range=lat_range,
                                     lon_range=lon_range)
    if df is None or df.empty:
        return None
    # EUREF SINEX 各分析中心经纬度列序不一致, 一律用本地站表覆盖
    st = dl.load_station_list_from_file('euref')
    if st is not None and not st.empty:
        st = st.copy()
        st['_sid'] = st['station_id'].astype(str).str[:4].str.upper()
        cmap = st.drop_duplicates('_sid').set_index('_sid')
        key = df['station_id'].astype(str).str[:4].str.upper()
        for c in ('lat', 'lon', 'height'):
            if c in cmap.columns:
                df[c] = key.map(cmap[c])
        before = len(df)
        df = df.dropna(subset=['lat', 'lon'])
        if before > len(df):
            print(f'    移除 {before - len(df)} 条无坐标记录')
    df['source'] = 'euref'
    return df


def fetch_australia(dl, date, hours=None):
    files = dl.download_australia_ztd(date=date, region='gar1', hours=hours)
    if not files:
        return None
    df = dl.parse_australia_files_batch(files)
    if df is None or df.empty:
        return None
    df = _attach_coords(df, dl, 'australia')
    df['source'] = 'australia'
    return df


def fetch_ucar(dl, date, hours=None):
    """UCAR 只有 0/6/12/18 时次; hours=None 时取全部 4 个"""
    avail = [0, 6, 12, 18]
    use = avail if not hours else sorted(
        {min(avail, key=lambda x: min(abs(x - h), abs(x - h + 24),
                                      abs(x - h - 24))) for h in hours})
    files = dl.download_ucar_pwv(date=date, hours=use)
    if not files:
        return None
    df = dl.parse_ucar_pwv_files_batch(files)
    if df is None or df.empty:
        return None
    df = _attach_coords(df, dl, 'ucar_cosmic')
    df['source'] = 'ucar_cosmic'
    return df


def fetch_suominet(dl, date):
    files = dl.download_suominet_ztd(date=date)
    if not files:
        return None
    dfs = []
    for f in files:
        try:
            d = dl.parse_suominet_file(f)
            if d is not None and not d.empty:
                dfs.append(d)
        except Exception as e:
            print(f'    [WARN] 解析 {Path(f).name} 失败: {e}')
    if not dfs:
        return None
    df = pd.concat(dfs, ignore_index=True)
    df = _attach_coords(df, dl, 'suominet')
    df['source'] = 'suominet'
    return df

def fetch_igs_whu(dl, date, cfg, lat_range=None, lon_range=None, workers=4):
    """IGS 武汉: 逐站逐日下载。站点数 × 天数 = 请求数, 是全脚本最慢的一步。

    每个线程用自己的 GNSSDownloader 实例 (共享实例的 stats/FTP 连接会竞争)。
    """
    st = dl.load_station_list_from_file('igs_whu')
    if st is None or st.empty:
        print('    [WARN] IGS 站点列表为空')
        return None

    if lat_range and lon_range and {'lat', 'lon'} <= set(st.columns):
        st = st[(st['lat'] >= lat_range[0]) & (st['lat'] <= lat_range[1]) &
                (st['lon'] >= lon_range[0]) & (st['lon'] <= lon_range[1])]
    print(f'    IGS 站点: {len(st)} 个 (workers={workers})')
    if st.empty:
        return None

    rows = [r for _, r in st.iterrows()]
    out = []

    def one(station, worker_dl):
        try:
            fp = worker_dl.download_igs_whu_ztd(
                station_id=station['station_id'], date=date)
            if not fp:
                return None
            d = worker_dl.parse_igs_whu_file(fp)
            if d is None or d.empty:
                return None
            d['lat'] = station.get('lat', np.nan)
            d['lon'] = station.get('lon', np.nan)
            d['height'] = station.get('height', np.nan)
            d['station_id'] = station['station_id']
            return d
        except Exception:
            return None

    if workers <= 1:
        for s in tqdm(rows, desc='    IGS', leave=False):
            d = one(s, dl)
            if d is not None:
                out.append(d)
    else:
        # 每线程一个 downloader, 复用避免反复读 config
        local = {}

        def task(station):
            import threading
            tid = threading.get_ident()
            if tid not in local:
                local[tid] = GNSSDownloader(
                    config_path=cfg['config_path'], data_source='igs_whu',
                    era5_dir=cfg['era5_dir'],
                    station_list_dir=cfg['station_list_dir'],
                    save_dir=cfg['save_dir'])
            return one(station, local[tid])

        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(task, s): s for s in rows}
            for f in tqdm(as_completed(futs), total=len(futs),
                          desc='    IGS', leave=False):
                try:
                    d = f.result()
                    if d is not None:
                        out.append(d)
                except Exception:
                    pass

    if not out:
        return None
    df = pd.concat(out, ignore_index=True)
    df['source'] = 'igs_whu'
    return df

def compute_pwv(df, date_str, era5_dir):
    """ZTD → PWV (EGM96 正高 + ERA5 逐小时 T/P 垂直改正 + Bevis)。

    缺 ERA5 时**不删行**: pwv_mm 留 NaN, ztd_mm 照常输出。
    UCAR 自带的 pwv_mm 原样保留, 不重算。
    """
    if 'pwv_mm' not in df.columns:
        df['pwv_mm'] = np.nan
    if 'ztd_mm' not in df.columns:
        df['ztd_mm'] = np.nan

    need = df['ztd_mm'].notna() & df['pwv_mm'].isna()
    if not need.any():
        return df, 0

    era5_file = Path(era5_dir) / f'ERA5_Multi_{date_str}.nc'
    if not era5_file.exists():
        print(f'    [跳过PWV] 无 ERA5: {era5_file.name} (ZTD 仍保留)')
        return df, 0

    try:
        ds = xr.open_dataset(era5_file)
        if 'valid_time' not in ds.coords:
            ds.close()
            print('    [跳过PWV] ERA5 缺 valid_time 坐标')
            return df, 0
        t2m = ds['t2m'].load()
        sp = ds['sp'].load()
        e5_hours = pd.to_datetime(ds['valid_time'].values).hour
        ds.close()
    except Exception as e:
        print(f'    [跳过PWV] ERA5 加载失败: {e}')
        return df, 0

    orog = None
    oro_file = Path(era5_dir).parent.parent.parent / 'ERA5_orography.nc'
    if oro_file.exists():
        try:
            dso = xr.open_dataset(oro_file)
            orog = dso['orography'].squeeze().values
            dso.close()
        except Exception as e:
            print(f'    [WARN] orography 加载失败: {e}')

    sub = df.loc[need]
    lats = sub['lat'].values.astype(float)
    lons = sub['lon'].values.astype(float)
    ztd = sub['ztd_mm'].values.astype(float)
    h_ell = sub['height'].values.astype(float) if 'height' in sub.columns \
        else np.zeros(len(sub))

    h_orth, geoid_n = to_orthometric(lats, lons, h_ell)
    if geoid_n is not None and np.isfinite(geoid_n).any():
        print(f'    大地水准面改正 N: [{np.nanmin(geoid_n):.1f}, '
              f'{np.nanmax(geoid_n):.1f}] m')

    e5_lat = t2m.coords['latitude'].values
    e5_lon = t2m.coords['longitude'].values
    t_arr = t2m.values
    p_arr = sp.values

    obs_h = pd.to_datetime(sub['datetime'].values).hour
    ti = np.array([int(np.argmin(np.abs(e5_hours - h))) for h in obs_h])
    yi = np.array([int(np.argmin(np.abs(e5_lat - v))) for v in lats])
    # ERA5 经度可能是 0..360, 站点是 -180..180
    lons_q = np.where(e5_lon.max() > 180.0, np.mod(lons, 360.0), lons)
    xi = np.array([int(np.argmin(np.abs(e5_lon - v))) for v in lons_q])

    tv = t_arr[ti, yi, xi].astype(float)
    pv = p_arr[ti, yi, xi].astype(float)

    if orog is not None:
        e5_h = orog[yi, xi].astype(float)
        dh = h_orth - e5_h
        t0 = tv.copy()
        tv = tv - L_RATE * dh
        expo = G_ACC / (L_RATE * R_GAS / M_AIR)
        m = np.abs(dh) >= 1.0
        with np.errstate(invalid='ignore', divide='ignore'):
            pv = np.where(m, pv * (tv / t0) ** expo, pv)
        print(f'    高程差(正高-ERA5): [{np.nanmin(dh):.1f}, '
              f'{np.nanmax(dh):.1f}] m, 中位数={np.nanmedian(dh):.1f} m')

    p_hpa = pv / 100.0
    f_fac = (1.0 - 0.00266 * np.cos(2 * np.radians(lats))
             - 0.00028 * (h_orth / 1000.0))
    zhd = 2.2768 * p_hpa / f_fac
    zwd = ztd - zhd
    zwd = np.where(zwd < 0, np.nan, zwd)
    Tm = 70.2 + 0.72 * tv
    Pi = 1e6 / (1000.0 * 461.5 * (3.739e3 / Tm + 0.221))
    pwv = Pi * zwd

    # 物理合理性: 只接受 0<PWV<100 mm, 其余留 NaN (不删行)
    pwv = np.where((pwv > 0) & (pwv < 100), pwv, np.nan)
    df.loc[need, 'pwv_mm'] = pwv
    n_ok = int(np.isfinite(pwv).sum())
    print(f'    PWV 计算: {n_ok}/{len(pwv)} 条有效')
    return df, n_ok

def append_station_parts(df, out_dir, time_fmt=TIME_FMT):
    """按站点追加到 <站点>.csv.part (跨日期累积), 返回 (站点数, 记录数)"""
    if df is None or df.empty:
        return 0, 0
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    d = df.copy()
    if 'datetime' not in d.columns:
        print('    [WARN] 无 datetime 列, 跳过导出')
        return 0, 0
    d['datetime'] = pd.to_datetime(d['datetime'], errors='coerce')
    d = d.dropna(subset=['datetime'])
    for c in ('ztd_mm', 'pwv_mm'):
        if c not in d.columns:
            d[c] = np.nan
        d[c] = pd.to_numeric(d[c], errors='coerce')
    # ZTD 或 PWV 任一有效即保留
    d = d[d['ztd_mm'].notna() | d['pwv_mm'].notna()]
    if d.empty:
        return 0, 0

    n_sta = n_rec = 0
    for sid, g in d.groupby('station_id', sort=True):
        g = g.sort_values('datetime').drop_duplicates('datetime', keep='first')
        rec = pd.DataFrame({
            'datetime': g['datetime'].dt.strftime(time_fmt),
            'ztd_mm': g['ztd_mm'].round(3).values,
            'pwv_mm': g['pwv_mm'].round(3).values,
        })
        p = out_dir / f'{safe_station_name(sid)}.csv.part'
        rec.to_csv(p, mode='a', header=not p.exists(), index=False)
        n_sta += 1
        n_rec += len(rec)
    return n_sta, n_rec


def finalize(out_dir, time_fmt=TIME_FMT):
    """.part → <站点>_<起>_<止>.csv; 跨日期排序去重, 全空的 pwv_mm 列删掉"""
    out_dir = Path(out_dir)
    parts = sorted(out_dir.glob('*.csv.part'))
    if not parts:
        print('[WARN] 无 .part 文件, 没有数据可合并')
        return 0, 0

    print(f'\n合并站点文件: {len(parts)} 站')
    n_file = n_row = 0
    for p in tqdm(parts, desc='合并'):
        try:
            d = pd.read_csv(p, dtype={'datetime': str})
            if d.empty:
                p.unlink(); continue
            ts = pd.to_datetime(d['datetime'], format=time_fmt, errors='coerce')
            d = d.assign(_ts=ts).dropna(subset=['_ts'])
            d = d.sort_values('_ts').drop_duplicates('datetime', keep='first')
            if d.empty:
                p.unlink(); continue
            t0 = d['_ts'].iloc[0].strftime(time_fmt)
            t1 = d['_ts'].iloc[-1].strftime(time_fmt)
            d = d.drop(columns=['_ts'])
            # 该站从来没有 PWV → 不输出这一列
            if 'pwv_mm' in d.columns and d['pwv_mm'].isna().all():
                d = d.drop(columns=['pwv_mm'])
            d.to_csv(out_dir / f'{p.name[:-9]}_{t0}_{t1}.csv', index=False)
            p.unlink()
            n_file += 1
            n_row += len(d)
        except Exception as e:
            print(f'  [WARN] 合并 {p.name} 失败: {e}')
    print(f'合并完成: {n_file} 个站点文件, {n_row} 条记录')
    return n_file, n_row

def main():
    ap = argparse.ArgumentParser(
        description='按日期区间下载全部 GNSS 源, 输出按站点的 ZTD/PWV 时序 CSV',
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--start', required=True, help='起始日期 YYYYMMDD')
    ap.add_argument('--end', required=True, help='结束日期 YYYYMMDD (含)')
    ap.add_argument('--sources', default=','.join(DEFAULT_SOURCES),
                    help=f'逗号分隔; 可选 {ALL_SOURCES}; '
                         f'默认全部免认证源 (suominet 需显式指定)')
    ap.add_argument('--output_dir', default='data/raw/labels/GNSS_ZTD_stations')
    ap.add_argument('--raw_output_dir', default=None,
                    help='原始下载与缓存目录；默认写入 <output_dir>/_raw_download')
    ap.add_argument('--config_path', default='download_lables/config_v2.yaml')
    ap.add_argument('--era5_dir',
                    default='download_lables/data/raw/labels/ERA5')
    ap.add_argument('--station_list_dir', default='download_lables/GNSS_list')
    ap.add_argument('--lat_min', type=float, default=None)
    ap.add_argument('--lat_max', type=float, default=None)
    ap.add_argument('--lon_min', type=float, default=None)
    ap.add_argument('--lon_max', type=float, default=None)
    ap.add_argument('--igs_workers', type=int, default=4,
                    help='IGS 逐站下载并发数 (被限速就降到 1)')
    ap.add_argument('--no_pwv', action='store_true',
                    help='只输出 ZTD, 不算 PWV (UCAR 自带的仍保留)')
    ap.add_argument('--dedup', action='store_true',
                    help='跨源按小时去重。注意会把亚小时 ZTD 抽稀成逐小时, '
                         '默认关闭以保全时间分辨率')
    ap.add_argument('--overwrite', action='store_true',
                    help='忽略续传记录, 全部重下')
    args = ap.parse_args()

    srcs = [s.strip() for s in args.sources.split(',') if s.strip()]
    bad = [s for s in srcs if s not in ALL_SOURCES]
    if bad:
        print(f'[ERROR] 未知数据源 {bad}; 可选: {ALL_SOURCES}')
        return 1
    if not srcs:
        print('[ERROR] --sources 为空')
        return 1

    try:
        d0 = datetime.strptime(args.start, '%Y%m%d')
        d1 = datetime.strptime(args.end, '%Y%m%d')
    except ValueError as e:
        print(f'[ERROR] 日期格式应为 YYYYMMDD: {e}')
        return 1
    if d1 < d0:
        print('[ERROR] --end 早于 --start')
        return 1

    lat_range = lon_range = None
    if None not in (args.lat_min, args.lat_max):
        lat_range = (args.lat_min, args.lat_max)
    if None not in (args.lon_min, args.lon_max):
        lon_range = (args.lon_min, args.lon_max)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 续传记录: {date_str: [已完成的 source, ...]}
    mf = out_dir / '_done.json'
    done = {}
    if mf.exists() and not args.overwrite:
        try:
            raw = json.loads(mf.read_text(encoding='utf-8'))
            if isinstance(raw, dict):
                done = {k: list(v) for k, v in raw.items()}
        except Exception as e:
            print(f'[WARN] 续传记录读取失败, 从头开始: {e}')

    dates = list(daterange(d0, d1))
    print('=' * 64)
    print('GNSS 站点级 ZTD/PWV 下载')
    print('=' * 64)
    print(f'日期区间: {args.start} ~ {args.end}  ({len(dates)} 天)')
    print(f'数据源  : {srcs}')
    if any(s in AUTH_SOURCES for s in srcs):
        print(f'  注意: {sorted(set(srcs) & AUTH_SOURCES)} 需要认证, '
              f'未配好会自动跳过')
    print(f'空间范围: lat={lat_range or "全球"}  lon={lon_range or "全球"}')
    print(f'PWV     : {"不计算" if args.no_pwv else "有 ERA5 就算"}')
    print(f'跨源去重: {"开 (逐小时)" if args.dedup else "关 (保全分辨率)"}')
    print(f'输出目录: {out_dir}')
    print('=' * 64)

    cfg = {'config_path': args.config_path, 'era5_dir': args.era5_dir,
           'station_list_dir': args.station_list_dir,
           'save_dir': args.raw_output_dir or str(out_dir / '_raw_download')}
    tot_rec = tot_pwv = 0

    for d in dates:
        ds = d.strftime('%Y%m%d')
        todo = [s for s in srcs if s not in done.get(ds, [])]
        if not todo:
            print(f'\n[{ds}] 全部源已完成, 跳过')
            continue
        print(f'\n{"=" * 64}\n[{ds}] 待下载: {todo}\n{"=" * 64}')

        frames = []
        for s in todo:
            try:
                print(f'  下载 {s} ...')
                dl = GNSSDownloader(config_path=args.config_path,
                                    data_source=s, era5_dir=args.era5_dir,
                                    station_list_dir=args.station_list_dir,
                                    save_dir=cfg['save_dir'])
                if s == 'igs_whu':
                    df = fetch_igs_whu(dl, d, cfg, lat_range, lon_range,
                                       workers=args.igs_workers)
                elif s == 'euref':
                    df = fetch_euref(dl, d, lat_range, lon_range)
                elif s == 'australia':
                    df = fetch_australia(dl, d)
                elif s == 'ucar_cosmic':
                    df = fetch_ucar(dl, d)
                elif s == 'suominet':
                    df = fetch_suominet(dl, d)
                else:
                    df = None

                if df is None or df.empty:
                    print(f'    无数据')
                else:
                    print(f'    获取 {len(df)} 条')
                    frames.append(df)
                done.setdefault(ds, []).append(s)
            except Exception as e:
                print(f'    [错误] {s}: {type(e).__name__}: {e}')

        if not frames:
            mf.write_text(json.dumps(done, indent=1), encoding='utf-8')
            continue

        day = pd.concat(frames, ignore_index=True)
        # 统一 sigma 列名 (EUREF 用 ztd_std_mm, 其余用 sigma_mm)
        if 'ztd_std_mm' in day.columns:
            if 'sigma_mm' in day.columns:
                day['sigma_mm'] = day['sigma_mm'].fillna(day['ztd_std_mm'])
            else:
                day['sigma_mm'] = day['ztd_std_mm']
            day = day.drop(columns=['ztd_std_mm'])

        # 空间过滤 (给了范围才做)
        if lat_range and {'lat'} <= set(day.columns):
            day = day[(day['lat'] >= lat_range[0]) & (day['lat'] <= lat_range[1])]
        if lon_range and {'lon'} <= set(day.columns):
            day = day[(day['lon'] >= lon_range[0]) & (day['lon'] <= lon_range[1])]

        if args.dedup:
            try:
                from download_gnss_by_fy3g import deduplicate_gnss_data
                day = deduplicate_gnss_data(day)
            except Exception as e:
                print(f'  [WARN] 去重失败, 保留原始记录: {e}')

        if day.empty:
            print('  过滤后无数据')
            mf.write_text(json.dumps(done, indent=1), encoding='utf-8')
            continue

        if not args.no_pwv:
            day, n_pwv = compute_pwv(day, ds, args.era5_dir)
            tot_pwv += n_pwv

        n_sta, n_rec = append_station_parts(day, out_dir)
        tot_rec += n_rec
        print(f'  写出: {n_sta} 站 / {n_rec} 条')
        mf.write_text(json.dumps(done, indent=1), encoding='utf-8')

    n_file, n_row = finalize(out_dir)
    print(f'\n{"=" * 64}')
    print('完成')
    print(f'{"=" * 64}')
    print(f'累计写入: {tot_rec} 条 (PWV 有效 {tot_pwv} 条)')
    print(f'站点文件: {n_file} 个, 合并后 {n_row} 条')
    print(f'输出目录: {out_dir}')
    print(f'续传记录: {mf}  (--overwrite 可忽略)')
    return 0


if __name__ == '__main__':
    sys.exit(main())







