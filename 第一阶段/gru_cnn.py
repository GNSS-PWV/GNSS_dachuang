import glob
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import optuna
import xgboost
import warnings
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# 限制CPU线程数，防止占用过高（32核通常比64核更高效）
os.environ['OMP_NUM_THREADS'] = '32'
os.environ['MKL_NUM_THREADS'] = '32'
os.environ['OPENBLAS_NUM_THREADS'] = '32'
torch.set_num_threads(32)

# 防止CUDA内存碎片，提高内存利用率
os.environ['PYTORCH_ALLOC_CONF'] = 'expandable_segments:True'

# 设置随机种子以确保结果可重复
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# 确定设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")


def get_file_list(path, typ="*.txt"):
    """
    递归获取目录下所有匹配 typ 的文件（包含子文件夹）。
    例如：path="xg_data", typ="*.txt" 时，会找到 xg_data 及其子目录中的所有 txt 文件。
    """
    pattern = os.path.join(path, "**", typ)
    return sorted(glob.glob(pattern, recursive=True))


def plot_scatter(y_true, y_pred, save_path, title="Prediction vs True (PS)"):
    """
    绘制预测值 vs 真实值的散点图，并保存为图片。
    """
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()

    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, s=5, alpha=0.3)

    v_min = float(min(y_true.min(), y_pred.min()))
    v_max = float(max(y_true.max(), y_pred.max()))
    plt.plot([v_min, v_max], [v_min, v_max], "r--", linewidth=1)

    plt.xlabel("True PS")
    plt.ylabel("Predicted PS")
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()


def read_data(fp):
    try:
        df = pd.read_csv(fp, header=0, sep=None, engine='python')
        first_col = df.columns[0]
        if (first_col is None) or (str(first_col).strip() == '') or str(first_col).startswith('Unnamed'):
            df = df.rename(columns={first_col: 'TIME'})
    except Exception:
        column_names = ['TIME', 'YEAR', 'DOY', 'LAT', 'LON', 'ELV', 'TS', 'PS', 'WPS', 'ZWD', 'ZHD', 'ZTD', 'PWV', 'Tm']
        df = pd.read_csv(fp, header=None, names=column_names, sep=None, engine='python')

    cols_needed = ['TIME', 'YEAR', 'DOY', 'LAT', 'LON', 'ELV', 'TS', 'PS', 'WPS', 'Tm']  # 仅保留输入和目标列
    df = df[[c for c in cols_needed if c in df.columns]]
    return df


def prepare_data(file_list, sample_ratio=0.3):
    dfs = []
    for i, fp in enumerate(file_list):
        df = read_data(fp)
        # 过滤年份：只保留2014-2018年的数据，排除2019年
        if 'YEAR' in df.columns:
            df = df[(df['YEAR'] >= 2014) & (df['YEAR'] <= 2018)]
        df['station_id'] = i  # 添加站点标识
        dfs.append(df)

    data = pd.concat(dfs, ignore_index=True).dropna().reset_index(drop=True)

    # 按站点均匀抽稀：每个站点按相同比例抽稀，保持时序完整性
    print(f"原始数据量: {len(data)}")
    print(f"站点数量: {data['station_id'].nunique()}")

    sampled_dfs = []
    for station_id in data['station_id'].unique():
        station_data = data[data['station_id'] == station_id]
        # 对每个站点进行抽稀，保持时序顺序
        sampled = station_data.sample(frac=sample_ratio, random_state=42).sort_index()
        sampled_dfs.append(sampled)

    data = pd.concat(sampled_dfs, ignore_index=True)
    print(f"抽稀后数据量: {len(data)} (比例: {sample_ratio})")
    print(f"每个站点平均数据量: {len(data) / data['station_id'].nunique():.0f}")
    # 打印年份分布
    if 'YEAR' in data.columns:
        print("数据年份分布:")
        print(data['YEAR'].value_counts().sort_index())

    # 仅保留输入相关的特征工程（经纬度、高程、时间）
    if 'TIME' in data.columns:
        data['TIME'] = pd.to_datetime(data['TIME'], errors='coerce')
        data['hour'] = data['TIME'].dt.hour
        data['month'] = data['TIME'].dt.month
        data['day'] = data['TIME'].dt.day
        data['weekday'] = data['TIME'].dt.weekday
        data['season'] = pd.cut(data['month'], bins=[0, 3, 6, 9, 12], labels=['春', '夏', '秋', '冬']).astype(
            'category').cat.codes
        # 添加时间相关的三角函数特征
        data['hour_sin'] = np.sin(2 * np.pi * data['hour'] / 24)
        data['hour_cos'] = np.cos(2 * np.pi * data['hour'] / 24)
        data['month_sin'] = np.sin(2 * np.pi * data['month'] / 12)
        data['month_cos'] = np.cos(2 * np.pi * data['month'] / 12)
        data['day_sin'] = np.sin(2 * np.pi * data['day'] / 31)
        data['day_cos'] = np.cos(2 * np.pi * data['day'] / 31)
        data['weekday_sin'] = np.sin(2 * np.pi * data['weekday'] / 7)
        data['weekday_cos'] = np.cos(2 * np.pi * data['weekday'] / 7)

    # 按站点 + 高度 + 时间排序，保证时序一致
    sort_cols = []
    for col in ['station_id', 'ELV', 'TIME']:
        if col in data.columns:
            sort_cols.append(col)
    if sort_cols:
        data = data.sort_values(sort_cols).reset_index(drop=True)

    # 目标变量的滞后特征（作为时序输入补充），按站点 + 高度分组
    group_cols = [c for c in ['station_id', 'ELV'] if c in data.columns]

    for lag in [1, 2, 3, 6, 12]:
        data[f'PS_lag{lag}'] = data.groupby(group_cols)['PS'].shift(lag)

    # 为PS添加额外的特征工程
    print("为PS添加额外特征工程...")
    # PS的气压差特征（使用已按组计算的滞后量）
    data['PS_diff1'] = data['PS'] - data['PS_lag1']
    data['PS_diff2'] = data['PS_lag1'] - data['PS_lag2']
    data['PS_diff3'] = data['PS_lag2'] - data['PS_lag3']
    # PS与TS的交互特征
    if 'TS' in data.columns:
        data['PS_TS_ratio'] = data['PS'] / (data['TS'] + 1e-8)
        data['PS_TS_product'] = data['PS'] * data['TS']
    # 气压高度校正（考虑高程对气压的影响）
    # 简化的气压高度公式：P = P0 * exp(-g*h/(R*T))，这里使用近似
    if 'ELV' in data.columns:
        data['PS_elev_corrected'] = data['PS'] * np.exp(0.00012 * data['ELV'])
    # PS的移动平均特征（分组 rolling）
    ps_group = data.groupby(group_cols)['PS']
    for w in [3, 6, 12]:
        data[f'PS_ma{w}'] = ps_group.rolling(window=w, min_periods=1).mean().reset_index(level=group_cols,
                                                                                          drop=True)
    # PS的移动标准差特征
    for w in [3, 6]:
        data[f'PS_std{w}'] = ps_group.rolling(window=w, min_periods=1).std().reset_index(level=group_cols,
                                                                                          drop=True)

    # WPS和PS的交互特征（同样按站点 + 高度分组）
    if 'WPS' in data.columns:
        print("为WPS和PS添加交互特征...")
        wps_group = data.groupby(group_cols)['WPS']
        # 添加WPS滞后特征
        data['WPS_lag1'] = wps_group.shift(1)
        data['WPS_lag2'] = wps_group.shift(2)
        # 水汽压与总气压的比值
        data['WPS_PS_ratio'] = data['WPS'] / (data['PS'] + 1e-8)
        # 干空气压（总气压减去水汽压）
        data['WPS_PS_diff'] = data['PS'] - data['WPS']
        # 乘积特征
        data['WPS_PS_product'] = data['WPS'] * data['PS']
        # WPS的移动平均特征
        for w in [3, 6]:
            data[f'WPS_ma{w}'] = wps_group.rolling(window=w, min_periods=1).mean().reset_index(level=group_cols,
                                                                                                drop=True)
    else:
        print("警告：数据中缺少WPS字段，跳过WPS相关特征")

    # TS的特征工程（按站点 + 高度分组）
    if 'TS' in data.columns:
        ts_group = data.groupby(group_cols)['TS']
        data['TS_lag1'] = ts_group.shift(1)
        data['TS_lag2'] = ts_group.shift(2)
        data['TS_diff1'] = data['TS'] - data['TS_lag1']
        data['TS_diff2'] = data['TS_lag1'] - data['TS_lag2']
        for w in [3, 6]:
            data[f'TS_ma{w}'] = ts_group.rolling(window=w, min_periods=1).mean().reset_index(level=group_cols,
                                                                                              drop=True)
        data['TS_std3'] = ts_group.rolling(window=3, min_periods=1).std().reset_index(level=group_cols, drop=True)

    # 经纬度的三角函数编码（提升空间特征表达）
    data['lat_sin'] = np.sin(data['LAT'] * np.pi / 180)
    data['lat_cos'] = np.cos(data['LAT'] * np.pi / 180)
    data['lon_sin'] = np.sin(data['LON'] * np.pi / 180)
    data['lon_cos'] = np.cos(data['LON'] * np.pi / 180)

    # 地理位置特征
    data['elevation_normalized'] = (data['ELV'] - data['ELV'].mean()) / (data['ELV'].std() + 1e-8)
    data['distance_from_equator'] = np.abs(data['LAT'])

    data = data.dropna()

    return data


def create_sequences(df, time_steps=24):
    """
    按“站点 + 高度”分组构造时序序列，保证一个序列内部是同一站点、同一高度随时间演变。
    返回：X, y, seasons, times, station_ids
      - X: (N, time_steps, feature_dim)
      - y: (N, 1)
      - seasons: 每个样本对应的季节编码（如果存在）
      - times:   每个样本目标时刻的时间戳（用于按时间/站点切分）
      - station_ids: 每个样本对应的站点编号（用于留站点验证）
    """
    # 仅保留输入相关特征：经纬度、高程、时间特征、目标滞后特征
    feats = [
        'LAT', 'LON', 'ELV',
        'hour', 'month', 'day', 'weekday', 'season',
        'hour_sin', 'hour_cos', 'month_sin', 'month_cos', 'day_sin', 'day_cos', 'weekday_sin', 'weekday_cos',
        'PS_lag1', 'PS_lag2', 'PS_lag3', 'PS_lag6', 'PS_lag12', 'lat_sin', 'lat_cos', 'lon_sin', 'lon_cos',
        'elevation_normalized', 'distance_from_equator',
        # 新增PS特征
        'PS_diff1', 'PS_diff2', 'PS_diff3', 'PS_TS_ratio', 'PS_TS_product', 'PS_elev_corrected', 'PS_ma3', 'PS_ma6',
        'PS_ma12', 'PS_std3', 'PS_std6',
        # 新增WPS相关特征
        'WPS_lag1', 'WPS_lag2', 'WPS_PS_ratio', 'WPS_PS_diff', 'WPS_PS_product', 'WPS_ma3', 'WPS_ma6',
        # 新增TS相关特征
        'TS_lag1', 'TS_lag2', 'TS_diff1', 'TS_diff2', 'TS_ma3', 'TS_ma6', 'TS_std3'
    ]
    feats = [f for f in feats if f in df.columns]
    target = ['PS']  # 只关注PS

    X, y, seasons, times, station_ids, years = [], [], [], [], [], []

    # 分组列：站点 + 高度，如果不存在就退化为不分组
    group_cols = [c for c in ['station_id', 'ELV'] if c in df.columns]
    if group_cols:
        grouped = df.groupby(group_cols)
    else:
        grouped = [(None, df)]

    for key, g in grouped:
        # 每个组内部按时间排序，保证是真实的时间序列
        if 'TIME' in g.columns:
            g = g.sort_values('TIME')
            time_vals = pd.to_datetime(g['TIME'], errors='coerce').values
        else:
            # 若不存在TIME列，则用索引代替时间顺序
            time_vals = np.arange(len(g))

        # 当前组对应的站点编号（若无station_id则记为-1）
        if 'station_id' in g.columns:
            current_station = int(g['station_id'].iloc[0])
        else:
            current_station = -1

        data_x = g[feats].values
        data_y = g[target].values
        season_vals = g['season'].values if 'season' in g.columns else None
        year_vals = g['YEAR'].values if 'YEAR' in g.columns else None

        if len(g) <= time_steps:
            continue

        for i in range(len(g) - time_steps):
            X.append(data_x[i:i + time_steps])
            y.append(data_y[i + time_steps])
            times.append(time_vals[i + time_steps])
            station_ids.append(current_station)
            if season_vals is not None:
                seasons.append(season_vals[i + time_steps])
            if year_vals is not None:
                years.append(year_vals[i + time_steps])

    X = np.array(X)
    y = np.array(y)
    seasons = np.array(seasons) if len(seasons) == len(X) and len(seasons) > 0 else None
    times = np.array(times)
    station_ids = np.array(station_ids)
    years = np.array(years) if 'YEAR' in df.columns else None

    if len(X) == 0 or len(y) == 0:
        raise ValueError(f"序列生成失败：有效序列数为0，time_steps={time_steps}")
    return X, y, seasons, times, station_ids, years


def station_based_split(X, y, times, station_ids, test_station_ratio=0.2, val_ratio=0.25, random_state=42):
    """
    留站点验证切分：
    - 先按 station_id 划分出测试站点（整站作为测试集）
    - 对剩余站点的数据，再按时间顺序划分训练 / 验证集
    """
    if not (len(X) == len(y) == len(times) == len(station_ids)):
        raise ValueError("X, y, times, station_ids 长度不一致，无法进行留站点切分")

    station_ids = np.array(station_ids)
    unique_stations = np.unique(station_ids)

    rng = np.random.RandomState(random_state)
    shuffled_stations = rng.permutation(unique_stations)

    n_test_stations = max(1, int(len(unique_stations) * test_station_ratio))
    test_stations = shuffled_stations[:n_test_stations]

    test_mask = np.isin(station_ids, test_stations)

    # 测试集：整站数据
    X_test, y_test, times_test = X[test_mask], y[test_mask], times[test_mask]

    # 剩余站点用于训练+验证
    X_temp, y_temp, times_temp = X[~test_mask], y[~test_mask], times[~test_mask]

    # 在剩余站点中按时间顺序切分训练 / 验证
    sort_idx = np.argsort(times_temp)
    X_temp_sorted = X_temp[sort_idx]
    y_temp_sorted = y_temp[sort_idx]

    n_temp = len(X_temp_sorted)
    val_size = max(1, int(n_temp * val_ratio))
    val_start = n_temp - val_size

    X_train, y_train = X_temp_sorted[:val_start], y_temp_sorted[:val_start]
    X_val, y_val = X_temp_sorted[val_start:], y_temp_sorted[val_start:]

    return X_train, X_val, X_test, y_train, y_val, y_test, test_stations


def time_based_split(X, y, times, years=None, test_ratio=0.2, val_ratio=0.25):
    """
    按时间顺序切分序列（场景A：同一批站点，预测未来时间）：
    - 先按 times 排序
    - 最后 test_ratio 部分作为测试集（时间上最靠后的样本）
    - 在剩余部分中，最后 val_ratio 部分作为验证集，其余为训练集
    
    如果提供了years参数，则按年份进行拆分：
    - 2014-2015年数据作为训练集
    - 2016-2018年数据作为测试验证集
    """
    if not (len(X) == len(y) == len(times)):
        raise ValueError("X, y, times 长度不一致，无法按时间切分")

    sort_idx = np.argsort(times)
    X_sorted = X[sort_idx]
    y_sorted = y[sort_idx]
    
    if years is not None and len(years) == len(times):
        # 按年份进行拆分
        years_sorted = np.array(years)[sort_idx]
        
        # 2014-2015年作为训练集
        train_mask = (years_sorted >= 2014) & (years_sorted <= 2015)
        # 2016-2018年作为测试验证集
        test_val_mask = (years_sorted >= 2016) & (years_sorted <= 2018)
        
        X_train = X_sorted[train_mask]
        y_train = y_sorted[train_mask]
        
        X_test_val = X_sorted[test_val_mask]
        y_test_val = y_sorted[test_val_mask]
        
        # 在测试验证集中按比例划分验证集和测试集
        n_test_val = len(X_test_val)
        val_size = max(1, int(n_test_val * val_ratio))
        val_start = n_test_val - val_size
        
        X_val = X_test_val[:val_start]
        y_val = y_test_val[:val_start]
        X_test = X_test_val[val_start:]
        y_test = y_test_val[val_start:]
        
        print(f"按年份拆分：")
        print(f"训练集（2014-2015）: {len(X_train)} 样本")
        print(f"验证集（2016-2018）: {len(X_val)} 样本")
        print(f"测试集（2016-2018）: {len(X_test)} 样本")
    else:
        # 原始按比例拆分逻辑
        n = len(times)
        test_size = max(1, int(n * test_ratio))
        test_start = n - test_size

        X_temp, y_temp = X_sorted[:test_start], y_sorted[:test_start]
        X_test, y_test = X_sorted[test_start:], y_sorted[test_start:]

        # 在 temp 中按时间顺序再切出验证集
        n_temp = len(X_temp)
        val_size = max(1, int(n_temp * val_ratio))
        val_start = n_temp - val_size

        X_train, y_train = X_temp[:val_start], y_temp[:val_start]
        X_val, y_val = X_temp[val_start:], y_temp[val_start:]

    return X_train, X_val, X_test, y_train, y_val, y_test


def remove_outliers(df, cols, method='iqr', iqr_factor=1.5, n_std=3):
    if method == 'iqr':
        for col in cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - iqr_factor * IQR
            upper_bound = Q3 + iqr_factor * IQR
            df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
    elif method == 'std':
        for col in cols:
            mean = df[col].mean()
            std = df[col].std()
            df = df[(df[col] >= mean - n_std * std) & (df[col] <= mean + n_std * std)]
    return df


def augment_time_series(x, noise_level=0.01, scale_range=(0.95, 1.05), shift_range=(-1, 1)):
    """
    精简版时序数据增强：
    仅添加小幅高斯噪声 + 轻量随机dropout，避免过强扰动导致整体欠拟合。
    """
    x_aug = x.copy()

    # 小幅高斯噪声（特征已标准化，0.005 量级足够）
    if noise_level and noise_level > 0:
        noise = np.random.normal(0, noise_level, x.shape)
        x_aug = x_aug + noise

    # 轻量随机 dropout 掩码（概率&幅度都很小）
    if np.random.random() < 0.3:
        dropout_prob = 0.05
        mask = np.random.binomial(1, 1 - dropout_prob, size=x_aug.shape)
        x_aug = x_aug * mask

    return x_aug


class TimeSeriesDataset(Dataset):
    "PyTorch时序数据集，支持训练集数据增强"

    def __init__(self, X, y, augment=False, noise_level=0.005):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.augment = augment
        self.noise_level = noise_level

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx].numpy()
        y = self.y[idx].numpy()

        if self.augment:
            x = augment_time_series(x, noise_level=self.noise_level)

        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


class LSTMModel(nn.Module):
    "增强的LSTM模型（无注意力机制，输出PS）"

    def __init__(self, input_size, hidden_size=128, num_layers=3, output_size=1, time_steps=24, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.time_steps = time_steps

        # 输入层层归一化（不依赖batch size）
        self.input_ln = nn.LayerNorm(input_size)

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # 输出层层归一化
        self.ln = nn.LayerNorm(hidden_size)

        # 更复杂的全连接网络
        self.fc_net = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, output_size)
        )

    def forward(self, x):
        # 输入层层归一化
        batch_size, seq_len, input_dim = x.shape
        x = self.input_ln(x)

        lstm_out, _ = self.lstm(x)
        # 直接使用LSTM的最后一个时间步的输出
        out = self.ln(lstm_out[:, -1, :])
        out = self.fc_net(out)
        return out


class GRUModel(nn.Module):
    "增强的GRU模型（无注意力机制，输出PS）"

    def __init__(self, input_size, hidden_size=128, num_layers=3, output_size=1, time_steps=24, dropout=0.2):
        super(GRUModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.time_steps = time_steps

        # 输入层层归一化（不依赖batch size）
        self.input_ln = nn.LayerNorm(input_size)

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # 输出层层归一化
        self.ln = nn.LayerNorm(hidden_size)

        # 更复杂的全连接网络
        self.fc_net = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, output_size)
        )

    def forward(self, x):
        # 输入层层归一化
        batch_size, seq_len, input_dim = x.shape
        x = self.input_ln(x)

        gru_out, _ = self.gru(x)
        # 直接使用GRU的最后一个时间步的输出
        out = self.ln(gru_out[:, -1, :])
        out = self.fc_net(out)
        return out


class EnhancedAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads=2, dropout=0.3):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        self.W_q = nn.Linear(hidden_dim, hidden_dim)
        self.W_k = nn.Linear(hidden_dim, hidden_dim)
        self.W_v = nn.Linear(hidden_dim, hidden_dim)
        self.W_o = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        batch_size, seq_len, hidden_dim = x.shape

        # 线性变换
        q = self.W_q(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.W_k(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.W_v(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # 计算注意力权重
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # 加权求和
        attn_out = torch.matmul(attn_weights, v).transpose(1, 2).contiguous().view(batch_size, seq_len, hidden_dim)
        attn_out = self.W_o(attn_out)
        attn_out = self.dropout(attn_out)

        # 残差连接和层归一化
        out = self.layer_norm(x + attn_out)
        return out


class MultiHeadAttention(nn.Module):
    """更复杂的多头注意力机制"""

    def __init__(self, hidden_dim, num_heads=4, dropout=0.3):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        # 线性变换层
        self.q_linear = nn.Linear(hidden_dim, hidden_dim)
        self.k_linear = nn.Linear(hidden_dim, hidden_dim)
        self.v_linear = nn.Linear(hidden_dim, hidden_dim)
        self.out_linear = nn.Linear(hidden_dim, hidden_dim)

        # 层归一化和 dropout
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

        # 位置编码
        self.position_encoding = self._create_position_encoding(hidden_dim, 1000)

    def _create_position_encoding(self, d_model, max_len):
        """创建位置编码"""
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)  # (1, max_len, d_model)

    def forward(self, x):
        batch_size, seq_len, hidden_dim = x.shape

        # 添加位置编码
        if seq_len > self.position_encoding.shape[1]:
            self.position_encoding = self._create_position_encoding(hidden_dim, seq_len)
        pe = self.position_encoding[:, :seq_len, :].to(x.device)
        x = x + pe

        # 线性变换
        q = self.q_linear(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_linear(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_linear(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # 计算注意力分数
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)

        # 应用掩码（可选，用于防止未来信息泄露）
        mask = torch.tril(torch.ones(seq_len, seq_len)).to(x.device)
        attn_scores = attn_scores.masked_fill(mask == 0, -1e9)

        # 计算注意力权重
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # 加权求和
        attn_out = torch.matmul(attn_weights, v).transpose(1, 2).contiguous().view(batch_size, seq_len, hidden_dim)

        # 输出线性变换
        out = self.out_linear(attn_out)
        out = self.dropout(out)

        # 残差连接和层归一化
        out = self.layer_norm(x + out)

        return out


# ========== CNN-BiLSTM+注意力模型 ==========
class CNNBiLSTMAttentionModel(nn.Module):
    "深度优化的CNN-BiLSTM+注意力模型（输出PS）"

    def __init__(self, input_size, hidden_size=128, num_layers=3, output_size=1, time_steps=48, dropout=0.25,
                 num_heads=4):
        super().__init__()
        self.time_steps = time_steps

        # 改进的CNN特征提取器（使用GroupNorm替代BatchNorm，不依赖batch size）
        self.cnn = nn.Sequential(
            nn.Conv1d(input_size, 64, kernel_size=3, padding=1),
            nn.GroupNorm(8, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.GroupNorm(8, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.GroupNorm(8, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.MaxPool1d(kernel_size=2)
        )

        # 增强的BiLSTM
        self.lstm = nn.LSTM(
            input_size=256,  # 匹配CNN输出通道
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # 多层注意力机制
        self.attention = MultiHeadAttention(hidden_size * 2, num_heads=num_heads, dropout=0.3)

        # 增强的全连接层
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2 * 2, 256),  # 调整输入维度以匹配混合池化
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, output_size)
        )

    def forward(self, x):
        x_cnn = x.transpose(1, 2)
        x_cnn = self.cnn(x_cnn)
        x_lstm = x_cnn.transpose(1, 2)

        lstm_out, _ = self.lstm(x_lstm)
        attn_out = self.attention(lstm_out)

        # 混合池化策略：全局平均池化 + 全局最大池化
        avg_pool = torch.mean(attn_out, dim=1)
        max_pool = torch.max(attn_out, dim=1)[0]
        # 合并池化结果
        pooled = torch.cat([avg_pool, max_pool], dim=1)
        # 调整全连接层输入维度
        out = self.fc(pooled)

        return out


def optimize_model_params(X_train, y_train, X_val, y_val, input_size, time_steps, model_type, output_size, n_trials=20):
    """使用optuna搜索模型最佳超参数"""
    print(f"开始{model_type}模型超参数搜索，搜索轮数: {n_trials}", flush=True)

    def objective(trial):
        # 基础参数
        params = {
            'learning_rate': trial.suggest_float('learning_rate', 1e-6, 1e-2, log=True),
            'hidden_size': trial.suggest_int('hidden_size', 64, 256),
            # 减弱dropout和weight_decay的搜索范围，避免过强正则导致欠拟合
            'dropout': trial.suggest_float('dropout', 0.1, 0.5),
            'weight_decay': trial.suggest_float('weight_decay', 1e-7, 1e-4, log=True),
            'num_layers': trial.suggest_int('num_layers', 1, 4)
        }

        # 根据模型类型添加特定参数
        if 'attention' in model_type:
            params['num_heads'] = trial.suggest_int('num_heads', 2, 8)

        # 创建数据加载器用于评估
        batch_size = trial.suggest_categorical('batch_size', [8, 16, 32, 64, 128])

        # 创建临时数据集和加载器
        from torch.utils.data import TensorDataset
        train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                                      torch.tensor(y_train, dtype=torch.float32))
        val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32),
                                    torch.tensor(y_val, dtype=torch.float32))

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        # 创建并训练模型
        try:
            model = train_model(
                train_loader, val_loader,
                input_size=input_size,
                time_steps=time_steps,
                model_type=model_type,
                output_size=output_size,
                epochs=50,  # 搜索时使用中等轮数，增加训练充分性
                **params
            )

            # 评估模型
            model.eval()
            val_loss = 0.0
            # 使用与训练相同的损失函数
            loss_function = 'log_cosh'  # 与训练部分保持一致

            if loss_function == 'mse':
                criterion = nn.MSELoss()
            elif loss_function == 'mae':
                criterion = nn.L1Loss()
            elif loss_function == 'huber':
                criterion = nn.HuberLoss(delta=1.0)
            elif loss_function == 'smooth_l1':
                criterion = nn.SmoothL1Loss()
            elif loss_function == 'log_cosh':
                class LogCoshLoss(nn.Module):
                    def __init__(self):
                        super().__init__()

                    def forward(self, pred, target):
                        return torch.mean(torch.log(torch.cosh(pred - target)))

                criterion = LogCoshLoss()
            elif loss_function == 'mape':
                class MAPELoss(nn.Module):
                    def __init__(self):
                        super().__init__()

                    def forward(self, pred, target):
                        return torch.mean(torch.abs((pred - target) / (target + 1e-8))) * 100

                criterion = MAPELoss()
            else:
                criterion = nn.MSELoss()

            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    val_loss += loss.item() * batch_X.size(0)

            val_loss /= len(val_loader.dataset)
            return val_loss
        except Exception as e:
            print(f"超参数搜索出错: {e}")
            return float('inf')

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials)

    print(f"\n最佳超参数 ({model_type}):")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    print(f"最佳验证损失: {study.best_value:.4f}")

    return study.best_params


def train_model(train_loader, val_loader, input_size, time_steps, model_type='lstm', output_size=1, epochs=100,
                **kwargs):
    # 从kwargs中获取超参数，如未提供则使用默认值
    learning_rate = kwargs.get('learning_rate', 0.0005)
    hidden_size = kwargs.get('hidden_size', 96)
    dropout = kwargs.get('dropout', 0.3)
    # 默认weight_decay略微减小，避免过强L2约束导致整体欠拟合
    weight_decay = kwargs.get('weight_decay', 1e-5)
    num_layers = kwargs.get('num_layers', 2)
    num_heads = kwargs.get('num_heads', 4)

    if model_type == 'lstm':
        model = LSTMModel(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            output_size=output_size,
            time_steps=time_steps,
            dropout=dropout
        ).to(device)
        model_name = "LSTM"

    elif model_type == 'gru':
        model = GRUModel(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            output_size=output_size,
            time_steps=time_steps,
            dropout=dropout
        ).to(device)
        model_name = "GRU"

    elif model_type == 'cnn_bilstm_attention':
        # 确保hidden_size能被num_heads整除
        if hidden_size % num_heads != 0:
            # 调整hidden_size为最接近的能被num_heads整除的值
            hidden_size = (hidden_size // num_heads) * num_heads
            if hidden_size < num_heads:
                hidden_size = num_heads

        model = CNNBiLSTMAttentionModel(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            output_size=output_size,
            time_steps=time_steps,
            dropout=dropout,
            num_heads=num_heads
        ).to(device)
        model_name = "CNN-BiLSTM-Attention"
    else:
        raise ValueError(f"不支持的模型类型: {model_type}")

    # 损失函数选择
    loss_function = 'log_cosh'  # 可选: 'mse', 'mae', 'huber', 'smooth_l1', 'log_cosh', 'mape'

    if loss_function == 'mse':
        criterion = nn.MSELoss()
    elif loss_function == 'mae':
        criterion = nn.L1Loss()
    elif loss_function == 'huber':
        criterion = nn.HuberLoss(delta=1.0)  # delta值可根据PS的典型范围调整
    elif loss_function == 'smooth_l1':
        criterion = nn.SmoothL1Loss()
    elif loss_function == 'log_cosh':
        # 自定义Log-Cosh Loss
        class LogCoshLoss(nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, pred, target):
                return torch.mean(torch.log(torch.cosh(pred - target)))

        criterion = LogCoshLoss()
    elif loss_function == 'mape':
        # 自定义MAPE Loss
        class MAPELoss(nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, pred, target):
                return torch.mean(torch.abs((pred - target) / (target + 1e-8))) * 100

        criterion = MAPELoss()
    else:
        criterion = nn.MSELoss()

    # 使用AdamW优化器，它在Adam的基础上添加了权重衰减正则化
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # 使用更先进的学习率调度策略
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-8
    )

    best_val_loss = float('inf')
    patience = 15
    patience_counter = 0
    best_model_state = None

    print(f"开始训练，总轮数: {epochs}, 早停耐心值: {patience}", flush=True)
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        batch_count = 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item() * batch_X.size(0)
            batch_count += 1
        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item() * batch_X.size(0)
        val_loss /= len(val_loader.dataset)

        scheduler.step()

        # 每5个epoch打印一次，同时打印第一个epoch以便确认训练开始
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(
                f'Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, LR: {optimizer.param_groups[0]["lr"]:.6f}',
                flush=True
            )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'Early stopping at epoch {epoch + 1}', flush=True)
                break
    
    print(f"训练完成，最佳验证损失: {best_val_loss:.4f}", flush=True)

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    return model


def predict_with_lstm(model, data_loader):
    model.eval()
    predictions = []
    targets = []
    with torch.no_grad():
        for batch_X, batch_y in data_loader:
            batch_X = batch_X.to(device)
            outputs = model(batch_X)
            predictions.append(outputs.cpu().numpy())
            targets.append(batch_y.numpy())
    return np.vstack(predictions), np.vstack(targets)


def predict_in_batches(model, X, batch_size=32):
    """在批次中进行预测，避免CUDA内存不足"""
    model.eval()
    predictions = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            batch_X = X[i:i+batch_size]
            batch_tensor = torch.tensor(batch_X, dtype=torch.float32).to(device)
            batch_preds = model(batch_tensor).cpu().numpy()
            predictions.append(batch_preds)
    return np.vstack(predictions)


def save_predictions(output_dir, model_preds):
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, "model_preds.npy"), model_preds)
    print(f"预测结果已保存至 {output_dir}")


def evaluate_models(y_test, model, test_loader, y_train=None, train_loader=None, model_type="CNN-BiLSTM-Attention"):
    model_preds, _ = predict_with_lstm(model, test_loader)

    # 计算测试集指标
    test_metrics = {
        'mse': mean_squared_error(y_test, model_preds),
        'rmse': np.sqrt(mean_squared_error(y_test, model_preds)),
        'mae': mean_absolute_error(y_test, model_preds),
        'r2': r2_score(y_test, model_preds)
    }

    # 打印测试集评估结果
    print(f"\n=== {model_type}模型评估（标准化尺度）===")
    print(f"测试集指标:")
    print(f"  MSE: {test_metrics['mse']:.4f}, RMSE: {test_metrics['rmse']:.4f}, MAE: {test_metrics['mae']:.4f}, R2: {test_metrics['r2']:.4f}")

    # 如果提供了训练集数据，计算训练集指标
    train_metrics = None
    if y_train is not None and train_loader is not None:
        train_preds, _ = predict_with_lstm(model, train_loader)
        train_metrics = {
            'mse': mean_squared_error(y_train, train_preds),
            'rmse': np.sqrt(mean_squared_error(y_train, train_preds)),
            'mae': mean_absolute_error(y_train, train_preds),
            'r2': r2_score(y_train, train_preds)
        }
        print(f"训练集指标:")
        print(f"  MSE: {train_metrics['mse']:.4f}, RMSE: {train_metrics['rmse']:.4f}, MAE: {train_metrics['mae']:.4f}, R2: {train_metrics['r2']:.4f}")
        
        # 检查过拟合
        overfit_ratio = test_metrics['mse'] / train_metrics['mse']
        print(f"过拟合比例 (测试MSE/训练MSE): {overfit_ratio:.2f}")
        if overfit_ratio > 1.5:
            print("  ?? 警告：可能存在过拟合！")
        elif overfit_ratio < 0.8:
            print("  ?? 警告：可能存在欠拟合！")
        else:
            print("  ? 模型拟合良好")

    return {'test': test_metrics, 'train': train_metrics}, model_preds


def optimize_xgboost_params(X_train, y_train, X_val, y_val):
    """使用Optuna优化XGBoost超参数"""
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 300),
            'max_depth': trial.suggest_int('max_depth', 3, 8),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.15),
            'subsample': trial.suggest_float('subsample', 0.7, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 1.0),
            'gamma': trial.suggest_float('gamma', 0, 0.3),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 3),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 0.5),
            'reg_lambda': trial.suggest_float('reg_lambda', 1, 3),
            'eval_metric': 'rmse',
            'tree_method': 'hist',
            'nthread': 32,  # 限制XGBoost线程数（32核通常比64核更高效）
            'early_stopping_rounds': 30
        }
        model = xgboost.XGBRegressor(**params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        preds = model.predict(X_val)
        return np.sqrt(mean_squared_error(y_val, preds))

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=20, show_progress_bar=True)
    return study.best_params


def xgboost_refinement(X_train, y_train, X_val, y_val, model_preds_train, model_preds_val):
    """使用XGBoost对CNN+BiLSTM+Attention预测结果进行精细化修正"""
    print("\n" + "=" * 60)
    print("开始XGBoost精细化修正...")
    print("=" * 60)
    
    # 将3维数据展平为2维 (样本数, 时间步长*特征数)
    if len(X_train.shape) == 3:
        X_train_flat = X_train.reshape(X_train.shape[0], -1)
        X_val_flat = X_val.reshape(X_val.shape[0], -1)
    else:
        X_train_flat = X_train
        X_val_flat = X_val
    
    # 拼接输入特征+统计特征+模型预测结果
    X_train_stats = np.hstack([
        X_train_flat,
        np.mean(X_train_flat, axis=1).reshape(-1, 1),
        np.std(X_train_flat, axis=1).reshape(-1, 1),
        model_preds_train
    ])
    X_val_stats = np.hstack([
        X_val_flat,
        np.mean(X_val_flat, axis=1).reshape(-1, 1),
        np.std(X_val_flat, axis=1).reshape(-1, 1),
        model_preds_val
    ])

    models = []
    target_names = ['PS']
    for i in range(y_train.shape[1]):
        print(f"训练XGBoost模型 {i + 1}/{len(target_names)} - {target_names[i]}")
        best_params = optimize_xgboost_params(X_train_stats, y_train[:, i], X_val_stats, y_val[:, i])
        best_params['early_stopping_rounds'] = 30
        best_params['nthread'] = 32  # 限制XGBoost线程数（32核通常比64核更高效）
        model = xgboost.XGBRegressor(**best_params)
        model.fit(X_train_stats, y_train[:, i], eval_set=[(X_val_stats, y_val[:, i])], verbose=False)
        models.append(model)
        
        # 特征重要性输出
        feature_names = (
                [f'feat_{j}' for j in range(X_train_flat.shape[1])] +
                ['mean', 'std', 'model_pred']
        )
        importance = model.feature_importances_
        print("特征重要性:", sorted(zip(feature_names, importance), key=lambda x: -x[1])[:5])
    
    print("=" * 60)
    return models


def evaluate_xgboost_models(xgb_models, X_test_stats, y_test, target_names=['PS']):
    """评估XGBoost模型"""
    print("\n" + "=" * 60)
    print("XGBoost模型评估")
    print("=" * 60)
    
    xgb_metrics = {}
    xgb_preds_list = []
    
    for i, model in enumerate(xgb_models):
        target_name = target_names[i]
        preds = model.predict(X_test_stats)
        xgb_preds_list.append(preds)
        
        xgb_metrics[target_name] = {
            'mse': mean_squared_error(y_test[:, i], preds),
            'rmse': np.sqrt(mean_squared_error(y_test[:, i], preds)),
            'mae': mean_absolute_error(y_test[:, i], preds),
            'r2': r2_score(y_test[:, i], preds)
        }
        
        print(f"\n{target_name}:")
        print(f"  MSE: {xgb_metrics[target_name]['mse']:.4f}")
        print(f"  RMSE: {xgb_metrics[target_name]['rmse']:.4f}")
        print(f"  MAE: {xgb_metrics[target_name]['mae']:.4f}")
        print(f"  R2: {xgb_metrics[target_name]['r2']:.4f}")
    
    xgb_preds = np.column_stack(xgb_preds_list)
    print("=" * 60)
    
    return xgb_metrics, xgb_preds


def main():
    # 服务器路径（如在集群上运行时可切换回这一版）
    root_dir = r"/share/home/u23114/tj23114/packages/dachuang_pwv/PS/xg_data"
    output_dir = r"/share/home/u23114/tj23114/packages/dachuang_pwv/PS/xg_result"

    # 本地Windows路径：与当前代码所在的PS目录一致
    #root_dir = r"C:\Users\YuexuanWang\Desktop\PS\xg_data"
    #output_dir = r"C:\Users\YuexuanWang\Desktop\PS\xg_r"
    print("开始加载数据...", flush=True)
    try:
        file_list = get_file_list(root_dir, "*.txt")
        if not file_list:
            print("错误：未找到任何txt文件！", flush=True)
            return

        result_data = prepare_data(file_list, sample_ratio=1)
        if result_data is None or result_data.empty:
            print("无有效数据！", flush=True)
            return

        # 移除异常值（针对PS）
        print("移除异常值...", flush=True)
        outlier_cols = ['PS']
        result_data = remove_outliers(result_data, outlier_cols, method='iqr')

        # 数据类型转换
        numeric_cols = ['YEAR', 'DOY', 'LAT', 'LON', 'ELV', 'TS', 'PS', 'WPS', 'Tm',
                        'hour', 'month', 'day', 'weekday', 'season',
                        'hour_sin', 'hour_cos', 'month_sin', 'month_cos', 'day_sin', 'day_cos', 'weekday_sin',
                        'weekday_cos',
                        'PS_lag1', 'PS_lag2', 'PS_lag3', 'PS_lag6', 'PS_lag12', 'WPS_lag1', 'WPS_lag2', 'TS_lag1',
                        'TS_lag2',
                        'lat_sin', 'lat_cos', 'lon_sin', 'lon_cos',
                        'elevation_normalized', 'distance_from_equator',
                        'PS_diff1', 'PS_diff2', 'PS_diff3', 'PS_TS_ratio', 'PS_TS_product', 'PS_elev_corrected',
                        'PS_ma3', 'PS_ma6', 'PS_ma12', 'PS_std3', 'PS_std6',
                        'WPS_PS_ratio', 'WPS_PS_diff', 'WPS_PS_product', 'WPS_ma3', 'WPS_ma6',
                        'TS_diff1', 'TS_diff2', 'TS_ma3', 'TS_ma6', 'TS_std3']
        present_cols = [c for c in numeric_cols if c in result_data.columns]
        result_data[present_cols] = result_data[present_cols].astype(np.float32)

        # 固定时间步长，不再搜索（减小内存和计算量）
        time_steps = 4
        batch_size = 32  # 同时确保batch_size也有定义

        # 创建最终的时序序列（按站点+高度分组，保证时间连续性）
        X_seq, y_seq, seasons, times, station_ids, years = create_sequences(result_data, time_steps)
        feature_dim = X_seq.shape[2]
        target_names = ['PS']
        print(f"序列形状：X={X_seq.shape}, y={y_seq.shape}（目标：PS）", flush=True)

        # 场景A：按时间顺序划分训练 / 验证 / 测试集（同一批站点，预测未来时间）
        print("按时间划分训练 / 验证 / 测试集（场景A）...", flush=True)
        X_train, X_val, X_test, y_train, y_val, y_test = time_based_split(
            X_seq, y_seq, times, years=years, test_ratio=0.2, val_ratio=0.25
        )

        # 使用RobustScaler，它对异常值更鲁棒
        from sklearn.preprocessing import RobustScaler
        x_scaler = RobustScaler()

        # 特征标准化
        X_train_flat = X_train.reshape(-1, feature_dim)
        X_train_scaled_flat = x_scaler.fit_transform(X_train_flat)
        X_train_scaled = X_train_scaled_flat.reshape(X_train.shape)

        X_val_flat = X_val.reshape(-1, feature_dim)
        X_val_scaled_flat = x_scaler.transform(X_val_flat)
        X_val_scaled = X_val_scaled_flat.reshape(X_val.shape)

        X_test_flat = X_test.reshape(-1, feature_dim)
        X_test_scaled_flat = x_scaler.transform(X_test_flat)
        X_test_scaled = X_test_scaled_flat.reshape(X_test.shape)

        # 为PS使用标准化器
        scalers = {}
        y_train_scaled = np.zeros_like(y_train)
        y_val_scaled = np.zeros_like(y_val)
        y_test_scaled = np.zeros_like(y_test)

        print("标准化目标变量...", flush=True)
        for i, name in enumerate(target_names):
            scalers[name] = StandardScaler()
            y_train_scaled[:, i] = scalers[name].fit_transform(y_train[:, i].reshape(-1, 1)).flatten()
            y_val_scaled[:, i] = scalers[name].transform(y_val[:, i].reshape(-1, 1)).flatten()
            y_test_scaled[:, i] = scalers[name].transform(y_test[:, i].reshape(-1, 1)).flatten()
            print(f"  {name} 标准化完成：均值={scalers[name].mean_[0]:.2f}, 标准差={np.sqrt(scalers[name].var_[0]):.2f}",
                  flush=True)

        # 创建数据加载器
        train_dataset = TimeSeriesDataset(X_train_scaled, y_train_scaled, augment=True, noise_level=0.01)
        val_dataset = TimeSeriesDataset(X_val_scaled, y_val_scaled, augment=False)
        test_dataset = TimeSeriesDataset(X_test_scaled, y_test_scaled, augment=False)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)

        # 训练模型
        models = {}
        best_params = {}
        # 模型组合：GRU + CNN+biLSTM+attention
        for model_type in ['gru', 'cnn_bilstm_attention']:
            # 定义模型显示名称映射
            model_display_names = {
                'gru': 'GRU',
                'cnn_bilstm_attention': 'CNN-BiLSTM-Attention'
            }
            display_name = model_display_names[model_type]

            # 动态设置输出维度，根据目标变量的数量
            output_size = y_train_scaled.shape[1]

            print(f"\n{'=' * 60}")
            print(f"搜索 {display_name} 模型的最佳超参数...", flush=True)
            print(f"{'=' * 60}")

            # 搜索超参数
            best_params[model_type] = optimize_model_params(
                X_train_scaled, y_train_scaled,
                X_val_scaled, y_val_scaled,
                input_size=feature_dim,
                time_steps=time_steps,
                model_type=model_type,
                output_size=output_size,
                n_trials=20  # 减少搜索次数以加快训练
            )

            print(f"\n{'=' * 60}")
            print(f"使用最佳超参数训练 {display_name} 模型...", flush=True)
            print(f"{'=' * 60}", flush=True)
            print(f"最佳超参数: {best_params[model_type]}", flush=True)

            # 为增强模型增加训练轮数
            epochs = 150 if model_type == 'cnn_bilstm_attention' else 100
            print(f"训练轮数: {epochs}", flush=True)

            # 使用最佳批量大小
            batch_size = best_params[model_type].get('batch_size', 32)

            # 创建最终的数据加载器
            # 训练用数据：带增强、打乱
            train_dataset = TimeSeriesDataset(X_train_scaled, y_train_scaled, augment=True, noise_level=0.005)
            val_dataset = TimeSeriesDataset(X_val_scaled, y_val_scaled, augment=False)
            test_dataset = TimeSeriesDataset(X_test_scaled, y_test_scaled, augment=False)

            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size)
            test_loader = DataLoader(test_dataset, batch_size=batch_size)

            # 评估用训练集数据：不增强、不打乱，保证与 y_train_scaled 一一对应
            train_eval_dataset = TimeSeriesDataset(X_train_scaled, y_train_scaled, augment=False)
            train_eval_loader = DataLoader(train_eval_dataset, batch_size=batch_size, shuffle=False)

            print(f"创建{display_name}模型...", flush=True)
            print(f"模型参数: input_size={feature_dim}, time_steps={time_steps}, output_size={output_size}", flush=True)
            print(f"训练参数: epochs={epochs}, batch_size={batch_size}", flush=True)
            
            try:
                model = train_model(
                    train_loader, val_loader,
                    input_size=feature_dim,
                    time_steps=time_steps,
                    model_type=model_type,
                    output_size=output_size,  # 动态输出维度
                    epochs=epochs,
                    **best_params[model_type]
                )
                models[model_type] = model
                print(f"{display_name}模型训练完成！", flush=True)
            except Exception as e:
                print(f"训练{display_name}模型时出错: {e}", flush=True)
                import traceback
                traceback.print_exc()
                raise

            # 评估模型
            metrics, model_preds = evaluate_models(
                y_test_scaled,
                model, test_loader,
                y_train=y_train_scaled,
                train_loader=train_eval_loader,
                model_type=display_name
            )

            # 反标准化并保存结果
            def inverse_transform_all(y_scaled, scalers, target_names):
                y_original = np.zeros_like(y_scaled)
                for i, name in enumerate(target_names):
                    y_scaled_reshaped = y_scaled[:, i].reshape(-1, 1)
                    y_inverted = scalers[name].inverse_transform(y_scaled_reshaped).flatten()
                    y_original[:, i] = y_inverted
                return y_original

            model_preds_original = inverse_transform_all(model_preds, scalers, target_names)
            save_predictions(os.path.join(output_dir, model_type), model_preds_original)

            # 在原始尺度下评估PS的精度
            y_true_original = inverse_transform_all(y_test_scaled, scalers, target_names)
            y_train_original = inverse_transform_all(y_train_scaled, scalers, target_names)

            print(f"\n{'=' * 60}", flush=True)
            print(f"{display_name} 模型 - 原始尺度下PS的精度：", flush=True)
            print(f"{'=' * 60}", flush=True)

            # 测试集指标
            y_true = y_true_original.flatten()
            y_pred = model_preds_original.flatten()

            test_mae = mean_absolute_error(y_true, y_pred)
            test_mse = mean_squared_error(y_true, y_pred)
            test_rmse = np.sqrt(test_mse)
            test_r2 = r2_score(y_true, y_pred)
            test_mape = np.mean(np.abs((y_pred - y_true) / (y_true + 1e-8))) * 100

            print(f"测试集 PS:", flush=True)
            print(f"  MAE: {test_mae:.4f}", flush=True)
            print(f"  MSE: {test_mse:.4f}", flush=True)
            print(f"  RMSE: {test_rmse:.4f}", flush=True)
            print(f"  R2: {test_r2:.4f}", flush=True)
            print(f"  MAPE: {test_mape:.2f}%", flush=True)

            # ====== 简单基线：使用训练集平均值进行预测（对比用） ======
            baseline_pred = np.full_like(y_true, fill_value=y_train_original.mean())
            baseline_mae = mean_absolute_error(y_true, baseline_pred)
            baseline_mse = mean_squared_error(y_true, baseline_pred)
            baseline_rmse = np.sqrt(baseline_mse)
            baseline_r2 = r2_score(y_true, baseline_pred)

            print(f"\n简单基线（预测训练集PS均值） - 测试集 PS:", flush=True)
            print(f"  MAE: {baseline_mae:.4f}", flush=True)
            print(f"  MSE: {baseline_mse:.4f}", flush=True)
            print(f"  RMSE: {baseline_rmse:.4f}", flush=True)
            print(f"  R2: {baseline_r2:.4f}", flush=True)

            # 绘制深度模型测试集散点图
            scatter_path_cnn = os.path.join(
                output_dir, model_type, "scatter_cnn_bilstm_attention_test.png"
            )
            plot_scatter(
                y_true,
                y_pred,
                scatter_path_cnn,
                title=f"{display_name} Test PS",
            )

            # 训练集指标
            train_preds_original = inverse_transform_all(
                predict_with_lstm(model, train_eval_loader)[0], scalers, target_names
            )
            y_train_true = y_train_original.flatten()
            y_train_pred = train_preds_original.flatten()

            train_mae = mean_absolute_error(y_train_true, y_train_pred)
            train_mse = mean_squared_error(y_train_true, y_train_pred)
            train_rmse = np.sqrt(train_mse)
            train_r2 = r2_score(y_train_true, y_train_pred)
            train_mape = np.mean(np.abs((y_train_pred - y_train_true) / (y_train_true + 1e-8))) * 100

            print(f"\n训练集 PS:", flush=True)
            print(f"  MAE: {train_mae:.4f}", flush=True)
            print(f"  MSE: {train_mse:.4f}", flush=True)
            print(f"  RMSE: {train_rmse:.4f}", flush=True)
            print(f"  R2: {train_r2:.4f}", flush=True)
            print(f"  MAPE: {train_mape:.2f}%", flush=True)
            print(f"{'=' * 60}", flush=True)

            # 使用XGBoost对CNN+BiLSTM+Attention预测结果进行精细化修正
            print(f"\n{'=' * 60}", flush=True)
            print(f"使用XGBoost对{display_name}预测结果进行精细化修正...", flush=True)
            print(f"{'=' * 60}", flush=True)

            # 获取训练集和验证集的模型预测结果
            model_preds_train = predict_in_batches(model, X_train_scaled)
            model_preds_val = predict_in_batches(model, X_val_scaled)

            # 使用XGBoost进行精细化修正
            xgb_models = xgboost_refinement(
                X_train_scaled, y_train_scaled, 
                X_val_scaled, y_val_scaled, 
                model_preds_train, model_preds_val
            )

            # 准备测试集的统计特征
            # 将3维数据展平为2维
            if len(X_test_scaled.shape) == 3:
                X_test_flat = X_test_scaled.reshape(X_test_scaled.shape[0], -1)
            else:
                X_test_flat = X_test_scaled
            
            X_test_stats = np.hstack([
                X_test_flat,
                np.mean(X_test_flat, axis=1).reshape(-1, 1),
                np.std(X_test_flat, axis=1).reshape(-1, 1),
                model_preds
            ])

            # 评估XGBoost模型
            xgb_metrics, xgb_preds = evaluate_xgboost_models(xgb_models, X_test_stats, y_test_scaled, target_names)

            # 反标准化XGBoost预测结果
            xgb_preds_original = inverse_transform_all(xgb_preds, scalers, target_names)

            # 在原始尺度下评估XGBoost修正后的PS精度
            print(f"\n{'=' * 60}", flush=True)
            print(f"{display_name} + XGBoost 模型 - 原始尺度下PS的精度：", flush=True)
            print(f"{'=' * 60}", flush=True)

            y_true = y_true_original.flatten()
            y_pred_xgb = xgb_preds_original.flatten()

            mae_xgb = mean_absolute_error(y_true, y_pred_xgb)
            mse_xgb = mean_squared_error(y_true, y_pred_xgb)
            rmse_xgb = np.sqrt(mse_xgb)
            r2_xgb = r2_score(y_true, y_pred_xgb)
            mape_xgb = np.mean(np.abs((y_pred_xgb - y_true) / (y_true + 1e-8))) * 100

            print(f"测试集 PS:", flush=True)
            print(f"  MAE: {mae_xgb:.4f}", flush=True)
            print(f"  MSE: {mse_xgb:.4f}", flush=True)
            print(f"  RMSE: {rmse_xgb:.4f}", flush=True)
            print(f"  R2: {r2_xgb:.4f}", flush=True)
            print(f"  MAPE: {mape_xgb:.2f}%", flush=True)
            print(f"{'=' * 60}", flush=True)

            # 对比模型和XGBoost修正后的结果
            print(f"\n{'=' * 60}", flush=True)
            print(f"性能对比 ({display_name}):", flush=True)
            print(f"{'=' * 60}", flush=True)
            print(f"{'指标':<15} {display_name:<20} {display_name}+XGBoost{'':<5} {'改善':<15}", flush=True)
            print(f"{'-' * 60}", flush=True)
            print(f"{'MAE':<15} {test_mae:<20.4f} {mae_xgb:<20.4f} {(test_mae - mae_xgb)/test_mae*100:+.2f}%", flush=True)
            print(f"{'RMSE':<15} {test_rmse:<20.4f} {rmse_xgb:<20.4f} {(test_rmse - rmse_xgb)/test_rmse*100:+.2f}%", flush=True)
            print(f"{'R2':<15} {test_r2:<20.4f} {r2_xgb:<20.4f} {(r2_xgb - test_r2)/test_r2*100:+.2f}%", flush=True)
            print(f"{'=' * 60}", flush=True)

            # 保存XGBoost修正后的预测结果
            save_predictions(os.path.join(output_dir, f"{model_type}_xgboost"), xgb_preds_original)

            # 绘制 XGBoost 修正后测试集散点图
            scatter_path_xgb = os.path.join(
                output_dir, f"{model_type}_xgboost", "scatter_cnn_bilstm_attention_xgboost_test.png"
            )
            plot_scatter(
                y_true,
                y_pred_xgb,
                scatter_path_xgb,
                title=f"{display_name} + XGBoost Test PS",
            )

            # 保存模型
            torch.save({
                'model_state_dict': model.state_dict(),
                'x_scaler': x_scaler,
                'scalers': scalers,
                'feature_dim': feature_dim,
                'time_steps': time_steps
            }, os.path.join(output_dir, f"{model_type}_model.pth"))

        print("\n所有模型训练完成！", flush=True)

    except Exception as e:
        print(f"执行出错：{e}", flush=True)
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()