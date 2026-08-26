# -*- coding: utf-8 -*-
"""
第二阶段数据管线: ZWD -> PWV 直接映射模型的数据加载与预处理

核心思路:
  - 每个时间戳的探空观测构成一条"垂直廓线"(变长高度层序列)
  - Transformer 处理廓线序列, 学习大气垂直结构 -> 转换系数 Pi 的非线性映射
  - Pi = PWV / ZWD, 最终 PWV = Pi * ZWD, 摆脱传统 Tm 线性假设

数据格式 (与第一阶段一致):
  每行: TIME, YEAR, DOY, LAT, LON, ELV, TS, PS, WPS, ZWD, ZHD, ZTD, PWV, Tm
  同一时间戳的多行 = 不同高度层, ELV 递增
  地表层 = 最低 ELV 行, 其 ZWD 即 GNSS 可观测的湿延迟
"""
import os
import glob
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler

# 每个高度层的输入特征 (探空实测值)
LEVEL_FEATURES = ['ELV', 'TS', 'PS', 'WPS']
LEVEL_FEATURE_DIM = len(LEVEL_FEATURES)

# 全局特征 (地表 GNSS 可观测 + 地理/时间)
GLOBAL_FEATURE_NAMES = [
    'zwd_surface', 'lat_sin', 'lat_cos', 'lon_sin', 'lon_cos',
    'doy_sin', 'doy_cos', 'hour_sin', 'hour_cos',
]
GLOBAL_FEATURE_DIM = len(GLOBAL_FEATURE_NAMES)

COLUMN_NAMES = [
    'TIME', 'YEAR', 'DOY', 'LAT', 'LON', 'ELV',
    'TS', 'PS', 'WPS', 'ZWD', 'ZHD', 'ZTD', 'PWV', 'Tm',
]


def read_station_file(filepath):
    """
    读取单个站点文件, 按 TIME 分组构建廓线列表.

    返回 list[dict], 每个元素:
      levels      : (n_levels, 4) ndarray [ELV, TS, PS, WPS]
      heights     : (n_levels,)   ndarray  原始高度(用于位置编码)
      global_raw  : dict  原始全局特征(未编码)
      pwv_surface : float 地表 PWV 真值 (mm)
      zwd_surface : float 地表 ZWD (mm)
      station_id  : str
      time_str    : str
    """
    station_id = os.path.basename(filepath).split('_met')[0]

    try:
        df = pd.read_csv(filepath, header=0, sep=None, engine='python')
        first_col = df.columns[0]
        if str(first_col).startswith('Unnamed') or str(first_col).strip() == '':
            df = df.rename(columns={first_col: 'TIME'})
    except Exception:
        df = pd.read_csv(filepath, header=None, names=COLUMN_NAMES, sep=None, engine='python')
        return []

    cols = [c for c in COLUMN_NAMES if c in df.columns]
    df = df[cols].copy()

    for c in ['ELV', 'TS', 'PS', 'WPS', 'ZWD', 'PWV', 'Tm', 'LAT', 'LON', 'DOY', 'YEAR']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    df = df.dropna(subset=['ELV', 'TS', 'PS', 'WPS', 'ZWD', 'PWV', 'LAT', 'LON'])

    if 'TIME' not in df.columns:
        return []

    df['TIME'] = pd.to_datetime(df['TIME'], errors='coerce')
    df = df.dropna(subset=['TIME'])

    # 按时间戳分组, 每组内按高度排序
    profiles = []
    for ts, g in df.groupby('TIME', sort=True):
        g = g.sort_values('ELV').reset_index(drop=True)
        if len(g) < 2:
            continue

        # 地表层 = 最低 ELV 行
        surface = g.iloc[0]
        zwd_s = float(surface['ZWD'])
        pwv_s = float(surface['PWV'])
        if zwd_s < 1.0 or pwv_s < 0.1:
            continue

        levels = g[LEVEL_FEATURES].values.astype(np.float32)
        heights = g['ELV'].values.astype(np.float32)

        doy = float(g['DOY'].iloc[0]) if 'DOY' in g.columns else 1.0
        hour = float(g['TIME'].iloc[0].hour)
        tm_s = float(g['Tm'].iloc[0]) if 'Tm' in g.columns else float('nan')

        profiles.append({
            'levels': levels,
            'heights': heights,
            'global_raw': {
                'zwd_surface': zwd_s,
                'lat': float(g['LAT'].iloc[0]),
                'lon': float(g['LON'].iloc[0]),
                'doy': doy,
                'hour': hour,
            },
            'pwv_surface': pwv_s,
            'zwd_surface': zwd_s,
            'tm_surface': tm_s,
            'elv_surface': float(surface['ELV']),
            'station_id': station_id,
            'time_str': str(ts),
        })

    return profiles


def load_all_profiles(data_dir, file_pattern='*_met.txt', max_files=None, year_filter=None):
    """
    递归读取目录下所有站点文件, 返回廓线列表.

    参数:
      data_dir    : 数据目录
      file_pattern: 文件匹配模式
      max_files   : 最多读取的文件数 (None=全部), 用于本地快速测试
      year_filter : (start, end) 年份过滤, 如 (2014, 2018)
    """
    import datetime as _dt

    pattern = os.path.join(data_dir, '**', file_pattern)
    files = sorted(glob.glob(pattern, recursive=True))
    if not files:
        pattern2 = os.path.join(data_dir, file_pattern)
        files = sorted(glob.glob(pattern2))
    if max_files is not None:
        files = files[:max_files]

    print(f'找到 {len(files)} 个站点文件', flush=True)
    all_profiles = []
    for i, fp in enumerate(files):
        profs = read_station_file(fp)
        if year_filter is not None:
            lo, hi = year_filter
            profs = [p for p in profs
                     if lo <= _dt.datetime.fromisoformat(p['time_str'].replace(' ', 'T')[:19]).year <= hi]
        all_profiles.extend(profs)
        if (i + 1) % 50 == 0:
            print(f'  已读取 {i+1}/{len(files)} 个文件, 累计 {len(all_profiles)} 条廓线', flush=True)

    print(f'共加载 {len(all_profiles)} 条廓线', flush=True)
    return all_profiles


def _encode_global(global_raw):
    """将原始全局特征编码为模型输入向量."""
    lat = global_raw['lat']
    lon = global_raw['lon']
    doy = global_raw['doy']
    hour = global_raw['hour']
    zwd = global_raw['zwd_surface']

    return np.array([
        zwd,
        np.sin(lat * np.pi / 180.0),
        np.cos(lat * np.pi / 180.0),
        np.sin(lon * np.pi / 180.0),
        np.cos(lon * np.pi / 180.0),
        np.sin(2 * np.pi * doy / 365.0),
        np.cos(2 * np.pi * doy / 365.0),
        np.sin(2 * np.pi * hour / 24.0),
        np.cos(2 * np.pi * hour / 24.0),
    ], dtype=np.float32)


class ProfileDataset(Dataset):
    """
    变长廓线数据集.

    每条样本返回:
      levels       : (n_levels, LEVEL_FEATURE_DIM)  归一化后的层特征
      heights      : (n_levels,)                     归一化高度
      global_feat  : (GLOBAL_FEATURE_DIM,)           归一化后的全局特征
      pi           : float                            转换系数真值 = PWV/ZWD
      pwv          : float                            PWV 真值 (mm)
      zwd          : float                            地表 ZWD (mm)
      station_id   : str
    """

    def __init__(self, profiles, level_scaler=None, global_scaler=None,
                 height_mean=None, height_std=None, fit_scalers=False):
        self.profiles = profiles

        if fit_scalers:
            all_levels = np.vstack([p['levels'] for p in profiles])
            self.level_scaler = StandardScaler()
            self.level_scaler.fit(all_levels)

            all_globals = np.stack([_encode_global(p['global_raw']) for p in profiles])
            self.global_scaler = StandardScaler()
            self.global_scaler.fit(all_globals)

            all_heights = np.concatenate([p['heights'] for p in profiles])
            self.height_mean = float(all_heights.mean())
            self.height_std = float(all_heights.std()) + 1e-8
        else:
            self.level_scaler = level_scaler
            self.global_scaler = global_scaler
            self.height_mean = height_mean
            self.height_std = height_std

        self._processed = []
        for p in profiles:
            levels_norm = self.level_scaler.transform(p['levels']).astype(np.float32)
            heights_norm = ((p['heights'] - self.height_mean) / self.height_std).astype(np.float32)
            global_feat = self.global_scaler.transform(
                _encode_global(p['global_raw']).reshape(1, -1)
            )[0].astype(np.float32)

            zwd = p['zwd_surface']
            pwv = p['pwv_surface']
            pi = pwv / zwd

            self._processed.append({
                'levels': levels_norm,
                'heights': heights_norm,
                'global_feat': global_feat,
                'pi': np.float32(pi),
                'pwv': np.float32(pwv),
                'zwd': np.float32(zwd),
                'station_id': p['station_id'],
                'time_str': p['time_str'],
                'lat': np.float32(p['global_raw']['lat']),
                'lon': np.float32(p['global_raw']['lon']),
                'elv_surface': np.float32(p.get('elv_surface', p['levels'][0][0])),
                'tm_surface': np.float32(p.get('tm_surface', 280.0)),
            })

    def __len__(self):
        return len(self._processed)

    def __getitem__(self, idx):
        return self._processed[idx]

    def get_scalers(self):
        return {
            'level_scaler': self.level_scaler,
            'global_scaler': self.global_scaler,
            'height_mean': self.height_mean,
            'height_std': self.height_std,
        }


def collate_profiles(batch, max_len=None):
    """
    将变长廓线 batch 填充为统一长度.

    返回:
      levels_pad    : (B, max_len, F)
      heights_pad   : (B, max_len)
      global_feat   : (B, G)
      attention_mask: (B, max_len)  True=有效层
      pi            : (B,)
      pwv           : (B,)
      zwd           : (B,)
      station_ids   : list[str]
    """
    lengths = [len(item['levels']) for item in batch]
    if max_len is None:
        max_len = max(lengths)

    B = len(batch)
    F = LEVEL_FEATURE_DIM
    G = GLOBAL_FEATURE_DIM

    levels_pad = np.zeros((B, max_len, F), dtype=np.float32)
    heights_pad = np.zeros((B, max_len), dtype=np.float32)
    attention_mask = np.zeros((B, max_len), dtype=bool)
    global_feat = np.zeros((B, G), dtype=np.float32)
    pi = np.zeros(B, dtype=np.float32)
    pwv = np.zeros(B, dtype=np.float32)
    zwd = np.zeros(B, dtype=np.float32)
    station_ids = []
    times = []
    lats = np.zeros(B, dtype=np.float32)
    lons = np.zeros(B, dtype=np.float32)
    elvs = np.zeros(B, dtype=np.float32)
    tms = np.zeros(B, dtype=np.float32)

    for i, item in enumerate(batch):
        n = min(len(item['levels']), max_len)
        levels_pad[i, :n] = item['levels'][:n]
        heights_pad[i, :n] = item['heights'][:n]
        attention_mask[i, :n] = True
        global_feat[i] = item['global_feat']
        pi[i] = item['pi']
        pwv[i] = item['pwv']
        zwd[i] = item['zwd']
        station_ids.append(item['station_id'])
        times.append(item['time_str'])
        lats[i] = item['lat']
        lons[i] = item['lon']
        elvs[i] = item['elv_surface']
        tms[i] = item['tm_surface']

    return {
        'levels': torch.from_numpy(levels_pad),
        'heights': torch.from_numpy(heights_pad),
        'global_feat': torch.from_numpy(global_feat),
        'attention_mask': torch.from_numpy(attention_mask),
        'pi': torch.from_numpy(pi),
        'pwv': torch.from_numpy(pwv),
        'zwd': torch.from_numpy(zwd),
        'station_ids': station_ids,
        'times': times,
        'lats': torch.from_numpy(lats),
        'lons': torch.from_numpy(lons),
        'elvs': torch.from_numpy(elvs),
        'tms': torch.from_numpy(tms),
    }


def station_based_split(profiles, test_station_ratio=0.1, val_ratio=0.15, random_state=42,
                        test_stations=None):
    """
    留站点验证切分: 先按 station_id 划分测试站, 再按时间划分训练/验证.

    参数:
      test_stations: 若给定(集合/列表/文件路径), 则直接用这些站作为测试站,
                     否则按 test_station_ratio 随机抽取(seed=random_state).

    返回: train_profiles, val_profiles, test_profiles
    """
    station_ids = np.array([p['station_id'] for p in profiles])
    unique_stations = np.unique(station_ids)

    if test_stations is not None:
        if isinstance(test_stations, str) and os.path.exists(test_stations):
            with open(test_stations, 'r', encoding='utf-8') as f:
                lines = [ln.strip() for ln in f if ln.strip() and not ln.strip().startswith('#')]
            test_stations = set()
            for ln in lines:
                test_stations.add(ln.split()[0])
        test_stations = set(test_stations)
        # 只保留数据中存在的测试站
        test_stations = test_stations & set(unique_stations)
        if len(test_stations) == 0:
            raise ValueError('指定的测试站与数据中的站点无交集!')
        train_val_stations = set(unique_stations) - test_stations
    else:
        rng = np.random.RandomState(random_state)
        shuffled = rng.permutation(unique_stations)
        n_test = max(1, int(len(unique_stations) * test_station_ratio))
        test_stations = set(shuffled[:n_test])
        train_val_stations = set(shuffled[n_test:])

    test_profiles = [p for p in profiles if p['station_id'] in test_stations]
    train_val_profiles = [p for p in profiles if p['station_id'] in train_val_stations]

    train_val_profiles.sort(key=lambda p: p['time_str'])
    n_val = int(len(train_val_profiles) * val_ratio)
    val_profiles = train_val_profiles[-n_val:] if n_val > 0 else []
    train_profiles = train_val_profiles[:-n_val] if n_val > 0 else train_val_profiles

    return train_profiles, val_profiles, test_profiles


def prepare_data(data_dir, batch_size=64, max_len=30,
                 test_station_ratio=0.1, val_ratio=0.15,
                 random_state=42, max_files=None, num_workers=0,
                 test_stations=None):
    """
    完整数据准备: 加载 -> 切分 -> 归一化 -> DataLoader.

    参数:
      test_stations: 指定测试站(集合/列表或文件路径), 与第一阶段官方 36 测试站对齐时使用.

    返回: train_loader, val_loader, test_loader, scalers, info
    """
    from torch.utils.data import DataLoader

    profiles = load_all_profiles(data_dir, max_files=max_files)
    if len(profiles) == 0:
        raise ValueError('未加载到任何廓线数据')

    train_profiles, val_profiles, test_profiles = station_based_split(
        profiles, test_station_ratio, val_ratio, random_state, test_stations=test_stations
    )

    print(f'站点划分: train={len(train_profiles)} val={len(val_profiles)} test={len(test_profiles)}', flush=True)
    train_stations = set(p['station_id'] for p in train_profiles)
    test_stations_set = set(p['station_id'] for p in test_profiles)
    print(f'  训练站: {len(train_stations)}  测试站: {len(test_stations_set)}', flush=True)

    train_ds = ProfileDataset(train_profiles, fit_scalers=True)
    scalers = train_ds.get_scalers()

    val_ds = ProfileDataset(val_profiles, **scalers) if len(val_profiles) > 0 else None
    test_ds = ProfileDataset(test_profiles, **scalers)

    collate_fn = lambda batch: collate_profiles(batch, max_len=max_len)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              collate_fn=collate_fn, num_workers=num_workers,
                              pin_memory=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            collate_fn=collate_fn, num_workers=num_workers,
                            pin_memory=True) if val_ds is not None else None
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             collate_fn=collate_fn, num_workers=num_workers,
                             pin_memory=True)

    info = {
        'n_train': len(train_ds),
        'n_val': len(val_ds) if val_ds is not None else 0,
        'n_test': len(test_ds),
        'max_len': max_len,
        'train_stations': sorted(train_stations),
        'test_stations': sorted(test_stations_set),
    }
    return train_loader, val_loader, test_loader, scalers, info


if __name__ == '__main__':
    data_dir = r'D:\gnss水汽反演\第一阶段\xg_test'
    train_loader, val_loader, test_loader, scalers, info = prepare_data(
        data_dir, batch_size=32, max_len=30, test_station_ratio=0.2
    )
    print(f"\n数据管线测试通过")
    print(f"  train: {info['n_train']}  val: {info['n_val']}  test: {info['n_test']}")
    batch = next(iter(train_loader))
    print(f"  batch levels: {batch['levels'].shape}")
    print(f"  batch heights: {batch['heights'].shape}")
    print(f"  batch global: {batch['global_feat'].shape}")
    print(f"  batch mask: {batch['attention_mask'].shape}")
    print(f"  pi range: [{batch['pi'].min():.4f}, {batch['pi'].max():.4f}]")
    print(f"  pwv range: [{batch['pwv'].min():.2f}, {batch['pwv'].max():.2f}]")
    print(f"  zwd range: [{batch['zwd'].min():.2f}, {batch['zwd'].max():.2f}]")