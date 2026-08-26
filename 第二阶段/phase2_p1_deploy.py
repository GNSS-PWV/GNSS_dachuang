# -*- coding: utf-8 -*-
"""Strict replay-style Phase-1-to-Phase-2 deployment comparison.

Phase-2 ProfileTransformer inputs remain a vertical profile
[ELV, TS, PS, WPS].  In this deployment simulation, the Phase-1 GRU surface
predictions PS / TS / WPS replace or adjust the profile surface state.  Tm is
loaded and retained only as an auxiliary diagnostic: it is not a current
ProfileTransformer input and never determines sample eligibility.

Regimes (all evaluated as PWV = Pi * true GNSS ZWD against radiosonde PWV):
  real          : true radiosonde profile (oracle)
  real_surf_p1  : true profile with its surface PS/TS/WPS replaced by Phase-1
  clim          : nearest-station seasonal climatological profile
  clim_surf_p1  : climatology with surface PS/TS/WPS replaced by Phase-1
  clim_adj_p1   : climatology adjusted throughout using Phase-1 surface state
  gpt3          : GPT3 Tm baseline

TIME-bearing Phase-1 files use strict station+TIME matching.  Legacy files
without TIME alone may use the original same-DOY ordered fallback.
"""
import os, sys, glob, pickle, argparse, json, hashlib
import numpy as np
import pandas as pd
import datetime as dt
import matplotlib
matplotlib.use('Agg')
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

BASE = '/share/home/u23114/tj23114/packages/dachuang_pwv'
DATA = f'{BASE}/PS/xg_data'
GRD = f'{BASE}/gpt3_1/gpt3_1.grd'
SEASONS = {'DJF': 15, 'MAM': 105, 'JJA': 198, 'SON': 288}
K2P = 22.1; K3 = 3.739e5; RV = 461.495; RHO_W = 1000.0
BIN_WIDTH = 1000.0; MAX_HEIGHT = 30000.0


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def pi_from_tm(Tm):
    return 1e8 / (RHO_W * RV * (K3 / Tm + K2P))


def season_of(month):
    if month in (12, 1, 2): return 'DJF'
    if month in (3, 4, 5): return 'MAM'
    if month in (6, 7, 8): return 'JJA'
    return 'SON'


def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    p1 = np.deg2rad(lat1); p2 = np.deg2rad(lat2)
    dp = np.deg2rad(lat2 - lat1); dl = np.deg2rad(lon2 - lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def encode_global(zwd, lat, lon, doy, hour):
    return np.array([zwd,
                     np.sin(np.deg2rad(lat)), np.cos(np.deg2rad(lat)),
                     np.sin(np.deg2rad(lon)), np.cos(np.deg2rad(lon)),
                     np.sin(2 * np.pi * doy / 365.0), np.cos(2 * np.pi * doy / 365.0),
                     np.sin(2 * np.pi * hour / 24.0), np.cos(2 * np.pi * hour / 24.0)], dtype=np.float32)


def mean_profile(levels_list):
    """Return the equal-weight, per-level mean of TS/PS/WPS profiles."""
    if not levels_list:
        return None
    all_elv = np.unique(np.concatenate([lv[:, 0] for lv in levels_list]))
    sums = np.zeros((len(all_elv), 3), dtype=np.float64)
    counts = np.zeros((len(all_elv), 3), dtype=np.int64)
    for lv in levels_list:
        pos = np.searchsorted(all_elv, lv[:, 0])
        values = lv[:, 1:4]
        for col in range(3):
            valid = np.isfinite(values[:, col])
            np.add.at(sums[:, col], pos[valid], values[valid, col])
            np.add.at(counts[:, col], pos[valid], 1)
    ok = np.all(counts > 0, axis=1)
    if not np.any(ok):
        return None
    means = sums[ok] / counts[ok]
    return np.column_stack([all_elv[ok], means]).astype(np.float32)


def metrics(pred, true):
    pred = np.asarray(pred); true = np.asarray(true)
    return {'RMSE': float(np.sqrt(mean_squared_error(true, pred))),
            'MAE': float(mean_absolute_error(true, pred)),
            'R2': float(r2_score(true, pred)),
            'Bias': float(np.mean(pred - true)),
            'RelRMSE_pct': float(np.sqrt(mean_squared_error(true, pred)) / np.mean(np.abs(true)) * 100),
            'N': int(len(true))}


def p1_time_key(value):
    t = pd.to_datetime(value, errors='coerce')
    return None if pd.isna(t) else t.strftime('%Y-%m-%dT%H:%M:%S')


def load_p1_preds(pred_dir, year):
    """Load Phase-1 predictions with new/legacy format metadata.

    Files with a TIME column use station+TIME as their only valid matching key.
    DOY-order matching is retained solely for legacy files with no TIME column.
    """
    out = {}
    files = sorted(glob.glob(os.path.join(pred_dir, f'*_{year}', '*.txt')))
    for fp in files:
        sid = os.path.basename(fp)[:-4]
        d = pd.read_csv(fp, sep=r'\s+')
        d.columns = [c.strip() for c in d.columns]
        by_time, by_doy = {}, {}
        has_time = 'TIME' in d.columns
        for _, r in d.iterrows():
            pred = float(r['Predict'])
            if has_time:
                key = p1_time_key(r['TIME'])
                if key is not None:
                    by_time.setdefault(key, pred)
            else:
                by_doy.setdefault(int(r['DOY']), []).append(pred)
        out[sid] = {'by_time': by_time, 'by_doy': by_doy, 'has_time': has_time}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model_dir', default='result_aligned')
    ap.add_argument('--year', type=int, default=2017)
    ap.add_argument('--cache', default=f'{BASE}/phase2/result_grid/st_seasonal_cache.pkl')
    ap.add_argument('--test_stations', default=f'{BASE}/phase2/test_stations_official_36.txt')
    ap.add_argument('--k', type=int, default=5)
    ap.add_argument('--out', default='result_p1deploy')
    ap.add_argument('--p1_root', default=None, help='新版第一阶段预测根目录（PS/WPS/TS/Tm/year_YYYY）；默认读取历史分散目录')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'设备: {device}', flush=True)

    # ---- Phase 2 model + scalers ----
    from model import ProfileTransformer
    ckpt = torch.load(os.path.join(args.model_dir, 'best_model.pth'), map_location=device, weights_only=True)
    model = ProfileTransformer(
        d_model=ckpt.get('args', {}).get('d_model', 128),
        n_heads=ckpt.get('args', {}).get('n_heads', 8),
        n_layers=ckpt.get('args', {}).get('n_layers', 4),
        ff_dim=ckpt.get('args', {}).get('ff_dim', 512),
        dropout=ckpt.get('args', {}).get('dropout', 0.1),
        global_feat_dim=9).to(device)
    model.load_state_dict(ckpt['model_state_dict']); model.eval()
    with open(os.path.join(args.model_dir, 'scalers.pkl'), 'rb') as f:
        scalers = pickle.load(f)
    ls = scalers['level_scaler']; gs = scalers['global_scaler']
    h_mean = scalers['height_mean']; h_std = scalers['height_std']
    print(f'Phase2 模型: {args.model_dir}', flush=True)

    # ---- Phase 1 predictions ----
    if args.p1_root:
        p1_dirs = {kind: os.path.join(args.p1_root, kind) for kind in ['PS', 'TS', 'WPS', 'Tm']}
    else:
        p1_dirs = {
            'PS': f'{BASE}/PS/ps_result_2/gru/predictions/PS',
            'TS': f'{BASE}/Ts_Tm/ts_result/gru/predictions/TS',
            'WPS': f'{BASE}/WPS/wps_result_1/gru/predictions',
            'Tm': f'{BASE}/Ts_Tm/tm_result/gru/predictions/Tm',
        }
    p1 = {kind: load_p1_preds(path, args.year) for kind, path in p1_dirs.items()}
    driving_kinds = ['PS', 'TS', 'WPS']
    diagnostic_kinds = ['Tm']
    common = set(p1['PS']) & set(p1['TS']) & set(p1['WPS'])
    print(
        f'Phase-1 files: PS={len(p1["PS"])} TS={len(p1["TS"])} WPS={len(p1["WPS"])} '
        f'Tm(aux)={len(p1["Tm"])}; driving intersection (PS/TS/WPS)={len(common)}',
        flush=True,
    )
    print('Phase-2 driving parameters: PS/TS/WPS. Tm is auxiliary-only and is not a ProfileTransformer input.', flush=True)

    # ---- climatological cache ----
    with open(args.cache, 'rb') as f:
        st = pickle.load(f)
    test_ids = set()
    with open(args.test_stations, 'r', encoding='utf-8') as f:
        for ln in f:
            ln = ln.strip()
            if ln and not ln.startswith('#'):
                test_ids.add(ln.split()[0])
    train_ids = [s for s in st.keys() if s not in test_ids]
    covered_test = sorted(common & test_ids)
    missing_test = sorted(test_ids - common)
    print(
        f'Official test file coverage (PS/TS/WPS): {len(covered_test)}/{len(test_ids)}; '
        f'missing: {", ".join(missing_test) if missing_test else "none"}',
        flush=True,
    )
    for kind in driving_kinds + diagnostic_kinds:
        missing_kind = sorted(test_ids - set(p1[kind]))
        if missing_kind:
            role = 'driving' if kind in driving_kinds else 'auxiliary diagnostic only'
            print(f'  {kind} ({role}) missing official stations ({len(missing_kind)}): {", ".join(missing_kind)}', flush=True)
    tl = np.array([st[s]['lat'] for s in train_ids])
    tn = np.array([st[s]['lon'] for s in train_ids])
    print(f'气候态训练站数: {len(train_ids)}', flush=True)

    # ---- GPT3 ----
    from gpt3 import GPT3
    gpt3 = GPT3(GRD)

    records = []
    match_counts = {kind: {'precise': 0, 'doy_fallback': 0, 'unmatched': 0}
                    for kind in driving_kinds + diagnostic_kinds}
    station_audit_rows = []
    stations = sorted(common)
    for sid in stations:
        station_audit = {
            'station': sid, 'raw_time_count': 0, 'valid_met_time_count': 0,
            'effective_time_count': 0, 'skipped_missing_met_file': 0,
            'skipped_no_valid_met_time': 0, 'skipped_missing_required_p1': 0,
            'skipped_missing_PS': 0, 'skipped_missing_TS': 0, 'skipped_missing_WPS': 0,
            'skipped_short_profile': 0, 'tm_diagnostic_unmatched': 0,
        }
        for kind in driving_kinds + diagnostic_kinds:
            for status in ['precise', 'doy_fallback', 'unmatched']:
                station_audit[f'{kind.lower()}_{status}'] = 0
        fp = f'{DATA}/{args.year}/{sid}_met.txt'
        if not os.path.exists(fp):
            station_audit['skipped_missing_met_file'] = 1
            station_audit_rows.append(station_audit)
            continue
        df = pd.read_csv(fp, header=0, sep=None, engine='python')
        fc = df.columns[0]
        if str(fc).startswith('Unnamed') or str(fc).strip() == '':
            df = df.rename(columns={fc: 'TIME'})
        df['TIME'] = pd.to_datetime(df['TIME'], errors='coerce')
        station_audit['raw_time_count'] = int(df['TIME'].nunique())
        for c in ['ELV', 'TS', 'PS', 'WPS', 'ZWD', 'PWV', 'Tm', 'LAT', 'LON']:
            df[c] = pd.to_numeric(df[c], errors='coerce')
        df = df.dropna(subset=['ELV', 'TS', 'PS', 'WPS', 'ZWD', 'PWV', 'TIME'])
        station_audit['valid_met_time_count'] = int(df['TIME'].nunique())
        if df.empty:
            station_audit['skipped_no_valid_met_time'] = 1
            station_audit_rows.append(station_audit)
            continue
        df['doy'] = df['TIME'].dt.dayofyear
        lat = float(df['LAT'].iloc[0]); lon = float(df['LON'].iloc[0])

        # 每 DOY 的观测时间顺序
        # Legacy DOY-only files use this ordered time map. New TIME-bearing
        # prediction files use strict station+TIME matching and never fall back.
        doy_times = {}
        for observed_time in sorted(pd.Timestamp(value) for value in df['TIME'].unique()):
            doy_times.setdefault(observed_time.dayofyear, []).append(observed_time)

        # 气候态: 最近 K 训练站
        d = haversine(tl, tn, lat, lon)
        idx = np.argsort(d)[:args.k]
        lat0, lon0 = lat, lon
        for ts, sgroup in df.groupby(df['TIME']):
            doy = int(ts.dayofyear)
            times_today = doy_times.get(doy, [])
            doy_index = times_today.index(ts) if ts in times_today else 0
            def p1val(kind):
                m = p1[kind].get(sid)
                if not m:
                    return None, 'unmatched'
                if m['has_time']:
                    exact = m['by_time'].get(p1_time_key(ts))
                    if exact is not None and np.isfinite(exact):
                        return float(exact), 'precise'
                    # A new TIME-bearing file must never fall back to DOY order.
                    return None, 'unmatched'
                values = m['by_doy'].get(doy)
                if not values:
                    return None, 'unmatched'
                value = values[doy_index] if doy_index < len(values) else float(np.mean(values))
                if value is None or not np.isfinite(value):
                    return None, 'unmatched'
                return float(value), 'doy_fallback'

            p1_values = {}
            for kind in driving_kinds + diagnostic_kinds:
                value, status = p1val(kind)
                p1_values[kind] = value
                match_counts[kind][status] += 1
                station_audit[f'{kind.lower()}_{status}'] += 1
            missing_driving = [kind for kind in driving_kinds if p1_values[kind] is None]
            if missing_driving:
                station_audit['skipped_missing_required_p1'] += 1
                for kind in missing_driving:
                    station_audit[f'skipped_missing_{kind}'] += 1
                continue
            if p1_values['Tm'] is None:
                station_audit['tm_diagnostic_unmatched'] += 1
            ps_p1 = p1_values['PS']
            ts_p1 = p1_values['TS']
            wps_p1 = p1_values['WPS']
            tm_p1 = p1_values['Tm']
            sub = sgroup.sort_values('ELV').reset_index(drop=True)
            surf = sub.iloc[0]
            zwd = float(surf['ZWD']); pwv_t = float(surf['PWV']); elv = float(surf['ELV'])
            hour = float(ts.hour)
            season = season_of(ts.month)
            prof = sub[['ELV', 'TS', 'PS', 'WPS']].values.astype(np.float32)
            if len(prof) < 2:
                station_audit['skipped_short_profile'] += 1
                continue

            # 气候态廓线 (最近K站季节平均)
            levels_list = []
            for kk in idx:
                if season in st[train_ids[kk]]['season']:
                    lv, _ = st[train_ids[kk]]['season'][season]
                    if len(lv) > 0:
                        levels_list.append(lv)
            clim_prof = mean_profile(levels_list)

            def run_profile(prof_in):
                levels_n = ls.transform(prof_in).astype(np.float32)
                heights_n = ((prof_in[:, 0] - h_mean) / h_std).astype(np.float32)
                gf = gs.transform(encode_global(zwd, lat, lon, doy, hour).reshape(1, -1))[0].astype(np.float32)
                L = len(prof_in)
                with torch.no_grad():
                    pi = model(torch.from_numpy(levels_n).unsqueeze(0).to(device),
                               torch.from_numpy(heights_n).unsqueeze(0).to(device),
                               torch.from_numpy(gf).unsqueeze(0).to(device),
                               torch.ones(1, L, dtype=torch.bool, device=device)).item()
                return float(pi)

            # 真实廓线
            pi_real = run_profile(prof)
            # 真实廓线 + 地表层用 P1 替换
            prof_surf_p1 = prof.copy(); prof_surf_p1[0, 1:4] = [ts_p1, ps_p1, wps_p1]
            pi_real_surf_p1 = run_profile(prof_surf_p1)

            pi_clim = pi_clim_surf = pi_clim_adj = np.nan
            if clim_prof is not None:
                pi_clim = run_profile(clim_prof)
                prof_cs = clim_prof.copy(); prof_cs[0, 1:4] = [ts_p1, ps_p1, wps_p1]
                pi_clim_surf = run_profile(prof_cs)
                # 整体订正
                c0 = clim_prof[0]
                dts = ts_p1 - float(c0[1]); rps = ps_p1 / float(c0[2]) if float(c0[2]) > 1 else 1.0
                rwps = wps_p1 / float(c0[3]) if float(c0[3]) > 1e-3 else 1.0
                prof_ca = clim_prof.copy()
                prof_ca[:, 1] = prof_ca[:, 1] + dts
                prof_ca[:, 2] = prof_ca[:, 2] * rps
                prof_ca[:, 3] = prof_ca[:, 3] * rwps
                pi_clim_adj = run_profile(prof_ca)

            # GPT3 Tm
            _, _, tm_gpt3, _ = gpt3.compute(lat, lon, elv, args.year, doy, hour)
            tm_gpt3_v = float(np.asarray(tm_gpt3).item())
            pwv_gpt3 = pi_from_tm(tm_gpt3_v) * zwd

            records.append({
                'station': sid, 'time': str(ts), 'doy': doy, 'hour': hour, 'zwd': zwd,
                'pwv_true': pwv_t, 'ps_true': float(surf['PS']), 'ts_true': float(surf['TS']),
                'wps_true': float(surf['WPS']), 'tm_true': float(surf['Tm']) if not pd.isna(surf['Tm']) else np.nan,
                'ps_p1': ps_p1, 'ts_p1': ts_p1, 'wps_p1': wps_p1, 'tm_p1': tm_p1 if tm_p1 is not None else np.nan,
                'pwv_real': pi_real * zwd, 'pwv_real_surf_p1': pi_real_surf_p1 * zwd,
                'pwv_clim': pi_clim * zwd if not np.isnan(pi_clim) else np.nan,
                'pwv_clim_surf_p1': pi_clim_surf * zwd if not np.isnan(pi_clim_surf) else np.nan,
                'pwv_clim_adj_p1': pi_clim_adj * zwd if not np.isnan(pi_clim_adj) else np.nan,
                'pwv_gpt3': pwv_gpt3,
            })
            station_audit['effective_time_count'] += 1
        station_audit_rows.append(station_audit)
        print(f'  站 {sid} 完成, 累计 {len(records)}', flush=True)

    audit = {
        'year': args.year,
        'protocol': 'replay_stacking_A_true_history_features',
        'phase2_driving_parameters': driving_kinds,
        'tm_role': 'auxiliary diagnostic only; not a ProfileTransformer input and not an eligibility requirement',
        'file_station_counts': {kind: len(p1[kind]) for kind in driving_kinds + diagnostic_kinds},
        'driving_file_intersection_station_count': len(common),
        'official_test_file_coverage': {
            'covered_station_count': len(covered_test), 'official_station_count': len(test_ids),
            'covered_station_ids': covered_test, 'missing_station_ids': missing_test,
        },
        'match_counts': match_counts,
        'provenance': {
            'model_dir': os.path.abspath(args.model_dir),
            'model_checkpoint_sha256': sha256_file(os.path.join(args.model_dir, 'best_model.pth')),
            'scalers_sha256': sha256_file(os.path.join(args.model_dir, 'scalers.pkl')),
            'cache_path': os.path.abspath(args.cache),
            'cache_sha256': sha256_file(args.cache),
            'gpt3_grid_path': os.path.abspath(GRD),
            'gpt3_grid_sha256': sha256_file(GRD),
        },
        'station_audit': station_audit_rows,
        'summary': {
            'station_count_attempted': len(stations),
            'raw_time_count': int(sum(row['raw_time_count'] for row in station_audit_rows)),
            'valid_met_time_count': int(sum(row['valid_met_time_count'] for row in station_audit_rows)),
            'effective_time_count': int(sum(row['effective_time_count'] for row in station_audit_rows)),
            'records_written': len(records),
        },
    }
    audit_json = os.path.join(args.out, f'p1_match_audit_{args.year}.json')
    audit_csv = os.path.join(args.out, f'p1_match_audit_{args.year}_per_station.csv')
    with open(audit_json, 'w', encoding='utf-8') as f:
        json.dump(audit, f, ensure_ascii=False, indent=2)
    pd.DataFrame(station_audit_rows).to_csv(audit_csv, index=False)
    print(f'P1 match audit saved: {audit_json}; station audit: {audit_csv}', flush=True)

    result_df = pd.DataFrame(records)
    prediction_csv = os.path.join(args.out, f'deploy_p1_predictions_{args.year}.csv')
    metrics_csv = os.path.join(args.out, f'deploy_p1_metrics_{args.year}.csv')
    if result_df.empty:
        result_df.to_csv(prediction_csv, index=False)
        pd.DataFrame(columns=['model', 'N', 'RMSE', 'MAE', 'R2', 'Bias', 'RelRMSE_pct']).to_csv(metrics_csv, index=False)
        raise RuntimeError(f'No effective deployment samples; inspect {audit_json} and {audit_csv}.')
    print(f'\nMatched samples: {len(result_df)}, stations: {result_df["station"].nunique()}', flush=True)
    result_df.to_csv(prediction_csv, index=False, float_format='%.6f')
    # Each model is reported on exactly the same rows. The separate analyzer
    # performs the six-year micro/annual/station-macro aggregation.
    regime_columns = [f'pwv_{name}' for name in ['real', 'real_surf_p1', 'clim', 'clim_surf_p1', 'clim_adj_p1', 'gpt3']]
    common_mask = result_df['pwv_true'].notna() & result_df[regime_columns].notna().all(axis=1)
    if not common_mask.any():
        raise RuntimeError(f'No common valid samples across all regimes; inspect {audit_json} and the climatology cache.')
    common_df = result_df.loc[common_mask]
    print('\n===== Phase-1 -> Phase-2 deployment comparison (PWV RMSE, mm; common samples) =====')
    rows = []
    for name in ['real', 'real_surf_p1', 'clim', 'clim_surf_p1', 'clim_adj_p1', 'gpt3']:
        met = metrics(common_df[f'pwv_{name}'].values, common_df['pwv_true'].values)
        met['model'] = name
        rows.append(met)
    out = pd.DataFrame(rows)[['model', 'N', 'RMSE', 'MAE', 'R2', 'Bias', 'RelRMSE_pct']]
    print(out.to_string(index=False))
    out.to_csv(metrics_csv, index=False, float_format='%.6f')
    print('saved ->', args.out)


if __name__ == '__main__':
    main()