# -*- coding: utf-8 -*-
"""
第一阶段 GRU 全时段推理（堆叠验证口径 A）
================================================
加载已训练的 Ps / Ts / WPS / Tm GRU，不重新训练；严格复用各模型训练脚本中的
预处理、特征列、序列构造与 scaler。输出只保留每站每时刻的最低高程（地表）记录，
供 phase2_p1_deploy.py 用 TIME 精确对齐。

注意：第一阶段原模型的特征含真值历史滞后项。本脚本用于“第一阶段输出驱动
第二阶段”的回放式堆叠验证，不等同于完全自治的实时递推部署。
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import torch

os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

BASE = Path('/share/home/u23114/tj23114/packages/dachuang_pwv')
DATA_ROOT = BASE / 'PS' / 'xg_data'

SPECS = {
    'PS': {
        'source': BASE / 'PS' / 'gru_ps_2.py',
        'checkpoint': BASE / 'PS' / 'ps_result_2' / 'gru' / 'gru_model.pth',
        'target': 'PS',
    },
    'WPS': {
        'source': BASE / 'WPS' / 'gru_wps_1.py',
        'checkpoint': BASE / 'WPS' / 'wps_result_1' / 'gru' / 'gru_model.pth',
        'target': 'WPS',
    },
    'TS': {
        'source': BASE / 'TS_TM' / 'gru_ts_fixed.py',
        'checkpoint': BASE / 'Ts_Tm' / 'ts_result' / 'gru' / 'gru_model.pth',
        'target': 'TS',
    },
    'Tm': {
        'source': BASE / 'TS_TM' / 'gru_tm_fixed.py',
        'checkpoint': BASE / 'Ts_Tm' / 'tm_result' / 'gru' / 'gru_model.pth',
        'target': 'Tm',
    },
}


def read_stations(path: Path | None) -> set[str] | None:
    if path is None:
        return None
    stations: set[str] = set()
    for line in path.read_text(encoding='utf-8').splitlines():
        line = line.strip()
        if line and not line.startswith('#'):
            stations.add(line.split()[0])
    if not stations:
        raise ValueError(f'站点文件没有有效站点：{path}')
    return stations


def load_legacy_module(source_path: Path, min_year: int, max_year: int) -> SimpleNamespace:
    """加载训练脚本，同时仅将其固定的 2014–2018 数据筛选替换为所需全时段。"""
    if not source_path.exists():
        raise FileNotFoundError(f'缺少训练脚本：{source_path}')
    source = source_path.read_text(encoding='utf-8')
    old = "df = df[(df['YEAR'] >= 2014) & (df['YEAR'] <= 2018)]"
    if old not in source:
        raise RuntimeError(f'未在 {source_path} 找到预期的年份筛选，拒绝猜测性修改。')
    source = source.replace(
        old,
        "df = df[(df['YEAR'] >= _P1_MIN_YEAR) & (df['YEAR'] <= _P1_MAX_YEAR)]",
        1,
    )
    ns: dict[str, Any] = {
        '__name__': f'_p1_legacy_{source_path.stem}',
        '__file__': str(source_path),
        '_P1_MIN_YEAR': min_year,
        '_P1_MAX_YEAR': max_year,
    }
    exec(compile(source, str(source_path), 'exec'), ns)
    required = ('prepare_data', 'remove_outliers', 'create_sequences', 'GRUModel')
    absent = [name for name in required if name not in ns]
    if absent:
        raise RuntimeError(f'{source_path} 缺少必要函数/模型：{absent}')
    return SimpleNamespace(**ns)


def all_input_files(min_year: int, max_year: int) -> list[str]:
    files: list[str] = []
    for year in range(min_year, max_year + 1):
        d = DATA_ROOT / str(year)
        if not d.is_dir():
            raise FileNotFoundError(f'缺少输入年份目录：{d}')
        files.extend(str(p) for p in sorted(d.glob('*_met.txt')))
    if not files:
        raise RuntimeError('未找到任何 *_met.txt 输入文件。')
    return files


def target_scaler(checkpoint: dict[str, Any], target: str):
    if 'scaler_y' in checkpoint:
        return checkpoint['scaler_y']
    scalers = checkpoint.get('scalers')
    if isinstance(scalers, dict) and target in scalers:
        return scalers[target]
    raise KeyError(f'checkpoint 未找到 {target} 的目标 scaler。keys={list(checkpoint.keys())}')


def build_model(legacy: SimpleNamespace, checkpoint: dict[str, Any], device: torch.device) -> torch.nn.Module:
    params = checkpoint.get('best_params', {})
    model = legacy.GRUModel(
        input_size=int(checkpoint['input_size']),
        hidden_size=int(params.get('hidden_size', 128)),
        num_layers=int(params.get('num_layers', 3)),
        output_size=1,
        time_steps=int(checkpoint['time_steps']),
        dropout=float(params.get('dropout', 0.2)),
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    model.eval()
    return model


def predict_batches(model: torch.nn.Module, x: np.ndarray, device: torch.device, batch_size: int) -> np.ndarray:
    chunks: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            xb = torch.from_numpy(x[start:start + batch_size]).to(device, non_blocking=True)
            chunks.append(model(xb).detach().cpu().numpy())
    return np.vstack(chunks) if chunks else np.empty((0, 1), dtype=np.float32)


def write_year_files(records: pd.DataFrame, kind: str, out_root: Path, years: list[int], overwrite: bool) -> dict[str, Any]:
    """写入 surface-only 的 TIME/DOY 预测文本；同一时间多高度时保留模型序列的首条（最低 ELV）。"""
    summary: dict[str, Any] = {}
    for year in years:
        year_dir = out_root / kind / f'year_{year}'
        part = records.loc[records['YEAR'] == year].copy()
        n_before = len(part)
        if part.empty:
            raise RuntimeError(f'[{kind} {year}] 没有可写出的预测序列；拒绝创建空年度目录。')
        # create_sequences 按 station_id、ELV 的 groupby 顺序输出，首条即最低 ELV 的有效序列。
        part = part.sort_values(['station', 'TIME', '_seq_order'], kind='stable').drop_duplicates(['station', 'TIME'], keep='first')
        if part.empty or part['station'].nunique() == 0:
            raise RuntimeError(f'[{kind} {year}] 地表去重后无有效预测；拒绝创建空年度目录。')
        duplicate_surface_keys = int(part.duplicated(['station', 'TIME']).sum())
        if duplicate_surface_keys:
            raise RuntimeError(f'[{kind} {year}] 地表去重后仍有 {duplicate_surface_keys} 个重复 (station, TIME) 键。')
        if year_dir.exists():
            if not overwrite:
                raise FileExistsError(f'输出目录已存在：{year_dir}（如需重写请添加 --overwrite）')
            shutil.rmtree(year_dir)
        year_dir.mkdir(parents=True, exist_ok=True)
        file_count = 0
        for sid, g in part.groupby('station', sort=True):
            g = g.sort_values('TIME', kind='stable')
            out = pd.DataFrame({
                'TIME': pd.to_datetime(g['TIME']).dt.strftime('%Y-%m-%dT%H:%M:%S'),
                'DOY': g['DOY'].astype(int),
                'StationID': sid,
                'Predict': g['Predict'].astype(float),
                'True': g['True'].astype(float),
                'Error': g['Error'].astype(float),
            })
            out.to_csv(year_dir / f'{sid}.txt', sep=' ', index=False, float_format='%.6f')
            file_count += 1
        summary[str(year)] = {
            'status': 'ok',
            'sequence_records_before_surface_dedup': int(n_before),
            'surface_records': int(len(part)),
            'stations': int(part['station'].nunique()),
            'files': file_count,
            'duplicate_surface_keys_after_dedup': duplicate_surface_keys,
        }
        print(f'[{kind} {year}] 序列记录 {n_before:,} -> 地表记录 {len(part):,}，站点 {file_count}', flush=True)
    return summary


def run_one(kind: str, args: argparse.Namespace, station_set: set[str] | None, device: torch.device) -> dict[str, Any]:
    spec = SPECS[kind]
    for p in (spec['source'], spec['checkpoint']):
        if not p.exists():
            raise FileNotFoundError(f'缺少 {kind} 所需文件：{p}')
    print(f'\n===== {kind}: 加载训练期预处理与 checkpoint =====', flush=True)
    legacy = load_legacy_module(spec['source'], args.data_min_year, args.data_max_year)
    checkpoint = torch.load(spec['checkpoint'], map_location='cpu', weights_only=False)
    expected_features = int(checkpoint['input_size'])
    time_steps = int(checkpoint['time_steps'])

    files = all_input_files(args.data_min_year, args.data_max_year)
    print(f'[{kind}] 输入文件 {len(files)}，特征构造年份 {args.data_min_year}–{args.data_max_year}', flush=True)
    data = legacy.prepare_data(files, sample_ratio=1)
    year_counts_before = {str(int(k)): int(v) for k, v in data['YEAR'].value_counts().sort_index().items()}
    if args.outlier_policy == 'legacy_global_iqr':
        # 仅用于复现旧脚本；阈值由完整输入期计算，不能当作严格在线部署预处理。
        data = legacy.remove_outliers(data, [spec['target']], method='iqr').reset_index(drop=True)
    elif args.outlier_policy != 'none':
        raise ValueError(f'未知 outlier_policy: {args.outlier_policy}')
    year_counts_after = {str(int(k)): int(v) for k, v in data['YEAR'].value_counts().sort_index().items()}
    if station_set is not None:
        data = data[data['station_id'].astype(str).isin(station_set)].copy()
    if data.empty:
        raise RuntimeError(f'[{kind}] 筛选后无有效数据。')
    print(f'[{kind}] 预处理后记录 {len(data):,}，站点 {data.station_id.nunique()}', flush=True)

    x, y, _seasons, times, stations, sample_years, doys = legacy.create_sequences(data, time_steps)
    if x.shape[2] != expected_features:
        raise RuntimeError(f'[{kind}] 特征维度不一致：重建={x.shape[2]}，checkpoint={expected_features}。')
    mask = np.isin(sample_years.astype(int), np.asarray(args.years, dtype=int))
    x = x[mask].astype(np.float32, copy=False)
    y = y[mask].astype(np.float64, copy=False)
    times = times[mask]
    stations = stations[mask].astype(str)
    sample_years = sample_years[mask].astype(int)
    doys = doys[mask].astype(int)
    if len(x) == 0:
        raise RuntimeError(f'[{kind}] 所选输出年份没有可用序列。')
    print(f'[{kind}] 待推理序列 {len(x):,}，时间步 {time_steps}，特征 {x.shape[2]}', flush=True)

    scaler_x = checkpoint.get('x_scaler', checkpoint.get('scaler_X'))
    if scaler_x is None:
        raise KeyError(f'[{kind}] checkpoint 无输入 scaler。')
    flat = x.reshape(-1, x.shape[-1])
    x_scaled = scaler_x.transform(flat).reshape(x.shape).astype(np.float32, copy=False)
    model = build_model(legacy, checkpoint, device)
    pred_scaled = predict_batches(model, x_scaled, device, args.batch_size)
    scaler_y = target_scaler(checkpoint, spec['target'])
    pred = scaler_y.inverse_transform(pred_scaled).reshape(-1)

    records = pd.DataFrame({
        'station': stations,
        'TIME': pd.to_datetime(times),
        'YEAR': sample_years,
        'DOY': doys,
        'Predict': pred,
        'True': y.reshape(-1),
        '_seq_order': np.arange(len(pred), dtype=np.int64),
    })
    records['Error'] = records['Predict'] - records['True']
    del x, x_scaled, flat, pred_scaled, model
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return {
        'target': spec['target'],
        'checkpoint': str(spec['checkpoint']),
        'input_records_before_outlier_policy_by_year': year_counts_before,
        'input_records_after_outlier_policy_by_year': year_counts_after,
        'outlier_policy': args.outlier_policy,
        'outputs': write_year_files(records, kind, args.out_root, args.years, args.overwrite),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description='第一阶段 GRU 2014–2019 全时段回放式推理')
    ap.add_argument('--years', nargs='+', type=int, default=[2014, 2015, 2016, 2017, 2018, 2019])
    ap.add_argument('--data-min-year', type=int, default=2014)
    ap.add_argument('--data-max-year', type=int, default=2019)
    ap.add_argument('--params', nargs='+', choices=list(SPECS), default=list(SPECS))
    ap.add_argument('--station-file', type=Path, default=BASE / 'phase2' / 'test_stations_official_36.txt',
                    help='只输出这些站；传空字符串不可用。默认官方36站。')
    ap.add_argument('--out-root', type=Path, default=BASE / 'phase2' / 'p1_full_predictions')
    ap.add_argument('--batch-size', type=int, default=2048)
    ap.add_argument('--outlier-policy', choices=['none', 'legacy_global_iqr'], default='none',
                    help='none 为严格部署推理默认值；legacy_global_iqr 仅用于复现旧脚本的全期 IQR 样本筛除。')
    ap.add_argument('--device', choices=['auto', 'cuda', 'cpu'], default='auto')
    ap.add_argument('--overwrite', action='store_true')
    args = ap.parse_args()
    args.years = sorted(set(args.years))
    if not args.years or min(args.years) < args.data_min_year or max(args.years) > args.data_max_year:
        raise ValueError('输出年份必须落在 data-min-year/data-max-year 之内。')
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('指定 --device cuda，但 CUDA 不可用。')
    device = torch.device('cuda' if args.device == 'auto' and torch.cuda.is_available() else args.device)
    stations = read_stations(args.station_file)
    print(f'设备: {device}; 参数: {args.params}; 输出年份: {args.years}; 输出站点: {len(stations) if stations else "全部"}', flush=True)
    args.out_root.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        'protocol': 'replay_stacking_A_true_history_features',
        'warning': 'Not autonomous recursive deployment; legacy features use historical observed meteorology.',
        'outlier_policy': args.outlier_policy,
        'outlier_policy_note': ('No target-based full-period filtering is applied.' if args.outlier_policy == 'none'
                                else 'Legacy global IQR filtering is enabled only for historical reproducibility; do not present it as strict online deployment.'),
        'data_years': [args.data_min_year, args.data_max_year],
        'prediction_years': args.years,
        'station_count_requested': len(stations) if stations else None,
        'params': {},
    }
    for kind in args.params:
        manifest['params'][kind] = run_one(kind, args, stations, device)
    (args.out_root / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'\n完成。结果根目录：{args.out_root}', flush=True)


if __name__ == '__main__':
    main()
