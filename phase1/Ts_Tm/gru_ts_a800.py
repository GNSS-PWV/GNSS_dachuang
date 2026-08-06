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
import warnings
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

os.environ['OMP_NUM_THREADS'] = '32'
os.environ['MKL_NUM_THREADS'] = '32'
os.environ['OPENBLAS_NUM_THREADS'] = '32'
torch.set_num_threads(32)

os.environ['PYTORCH_ALLOC_CONF'] = 'expandable_segments:True'

np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")


def get_file_list(path, typ="*.txt"):
    pattern = os.path.join(path, "**", typ)
    return sorted(glob.glob(pattern, recursive=True))


def plot_scatter(y_true, y_pred, save_path, title="Prediction vs True"):
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()

    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, s=5, alpha=0.3)

    v_min = float(min(y_true.min(), y_pred.min()))
    v_max = float(max(y_true.max(), y_pred.max()))
    plt.plot([v_min, v_max], [v_min, v_max], "r--", linewidth=1)

    plt.xlabel("True Value")
    plt.ylabel("Predicted Value")
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

    cols_needed = ['TIME', 'YEAR', 'DOY', 'LAT', 'LON', 'ELV', 'TS', 'PS', 'WPS', 'Tm']
    df = df[[c for c in cols_needed if c in df.columns]]
    return df


def prepare_data(file_list, sample_ratio=1):
    dfs = []
    for i, fp in enumerate(file_list):
        df = read_data(fp)
        if 'YEAR' in df.columns:
            df = df[(df['YEAR'] >= 2014) & (df['YEAR'] <= 2018)]
        filename = os.path.basename(fp)
        station_name = filename.split('_met')[0] if '_met' in filename else filename.split('.')[0]
        df['station_id'] = station_name
        dfs.append(df)

    data = pd.concat(dfs, ignore_index=True).dropna().reset_index(drop=True)

    print(f"数据量: {len(data)}")
    print(f"站点数量: {data['station_id'].nunique()}")
    if 'YEAR' in data.columns:
        print("数据年份分布:")
        print(data['YEAR'].value_counts().sort_index())

    if 'TIME' in data.columns:
        data['TIME'] = pd.to_datetime(data['TIME'], errors='coerce')
        data['hour'] = data['TIME'].dt.hour
        data['month'] = data['TIME'].dt.month
        data['day'] = data['TIME'].dt.day
        data['weekday'] = data['TIME'].dt.weekday
        data['season'] = pd.cut(data['month'], bins=[0, 3, 6, 9, 12], labels=['春', '夏', '秋', '冬']).astype(
            'category').cat.codes
        data['hour_sin'] = np.sin(2 * np.pi * data['hour'] / 24)
        data['hour_cos'] = np.cos(2 * np.pi * data['hour'] / 24)
        data['month_sin'] = np.sin(2 * np.pi * data['month'] / 12)
        data['month_cos'] = np.cos(2 * np.pi * data['month'] / 12)
        data['day_sin'] = np.sin(2 * np.pi * data['day'] / 31)
        data['day_cos'] = np.cos(2 * np.pi * data['day'] / 31)
        data['weekday_sin'] = np.sin(2 * np.pi * data['weekday'] / 7)
        data['weekday_cos'] = np.cos(2 * np.pi * data['weekday'] / 7)

    sort_cols = []
    for col in ['station_id', 'ELV', 'TIME']:
        if col in data.columns:
            sort_cols.append(col)
    if sort_cols:
        data = data.sort_values(sort_cols).reset_index(drop=True)

    group_cols = [c for c in ['station_id', 'ELV'] if c in data.columns]

    if 'TS' in data.columns:
        ts_group = data.groupby(group_cols)['TS']
        for lag in [1, 2, 3, 6, 12]:
            data[f'TS_lag{lag}'] = ts_group.shift(lag)
        data['TS_diff1'] = data['TS'] - data['TS_lag1']
        data['TS_diff2'] = data['TS_lag1'] - data['TS_lag2']
        data['TS_diff3'] = data['TS_lag2'] - data['TS_lag3']
        for w in [3, 6, 12]:
            data[f'TS_ma{w}'] = ts_group.rolling(window=w, min_periods=1).mean().reset_index(level=group_cols, drop=True)
        for w in [3, 6]:
            data[f'TS_std{w}'] = ts_group.rolling(window=w, min_periods=1).std().reset_index(level=group_cols, drop=True)

    if 'Tm' in data.columns:
        tm_group = data.groupby(group_cols)['Tm']
        for lag in [1, 2, 3, 6, 12]:
            data[f'Tm_lag{lag}'] = tm_group.shift(lag)
        data['Tm_diff1'] = data['Tm'] - data['Tm_lag1']
        data['Tm_diff2'] = data['Tm_lag1'] - data['Tm_lag2']
        data['Tm_diff3'] = data['Tm_lag2'] - data['Tm_lag3']
        for w in [3, 6, 12]:
            data[f'Tm_ma{w}'] = tm_group.rolling(window=w, min_periods=1).mean().reset_index(level=group_cols, drop=True)
        for w in [3, 6]:
            data[f'Tm_std{w}'] = tm_group.rolling(window=w, min_periods=1).std().reset_index(level=group_cols, drop=True)

    if 'PS' in data.columns:
        ps_group = data.groupby(group_cols)['PS']
        data['PS_lag1'] = ps_group.shift(1)
        data['PS_lag2'] = ps_group.shift(2)
        data['PS_diff1'] = data['PS'] - data['PS_lag1']
        data['PS_diff2'] = data['PS_lag1'] - data['PS_lag2']
        for w in [3, 6]:
            data[f'PS_ma{w}'] = ps_group.rolling(window=w, min_periods=1).mean().reset_index(level=group_cols, drop=True)
        data['PS_std3'] = ps_group.rolling(window=3, min_periods=1).std().reset_index(level=group_cols, drop=True)

    if 'WPS' in data.columns:
        wps_group = data.groupby(group_cols)['WPS']
        data['WPS_lag1'] = wps_group.shift(1)
        data['WPS_lag2'] = wps_group.shift(2)
        data['WPS_diff1'] = data['WPS'] - data['WPS_lag1']
        data['WPS_diff2'] = data['WPS_lag1'] - data['WPS_lag2']
        for w in [3, 6]:
            data[f'WPS_ma{w}'] = wps_group.rolling(window=w, min_periods=1).mean().reset_index(level=group_cols, drop=True)
        data['WPS_std3'] = wps_group.rolling(window=3, min_periods=1).std().reset_index(level=group_cols, drop=True)

    if 'TS' in data.columns and 'Tm' in data.columns:
        data['TS_Tm_ratio'] = data['TS'] / (data['Tm'] + 1e-8)
        data['TS_Tm_product'] = data['TS'] * data['Tm']
        data['TS_Tm_diff'] = data['TS'] - data['Tm']

    data['lat_sin'] = np.sin(data['LAT'] * np.pi / 180)
    data['lat_cos'] = np.cos(data['LAT'] * np.pi / 180)
    data['lon_sin'] = np.sin(data['LON'] * np.pi / 180)
    data['lon_cos'] = np.cos(data['LON'] * np.pi / 180)

    data['elevation_normalized'] = (data['ELV'] - data['ELV'].mean()) / (data['ELV'].std() + 1e-8)
    data['distance_from_equator'] = np.abs(data['LAT'])

    data = data.dropna()

    return data


def create_sequences(df, time_steps=24):
    feats = [
        'LAT', 'LON', 'ELV',
        'hour', 'month', 'day', 'weekday', 'season',
        'hour_sin', 'hour_cos', 'month_sin', 'month_cos', 'day_sin', 'day_cos', 'weekday_sin', 'weekday_cos',
        'lat_sin', 'lat_cos', 'lon_sin', 'lon_cos',
        'elevation_normalized', 'distance_from_equator',
        'TS_lag1', 'TS_lag2', 'TS_lag3', 'TS_lag6', 'TS_lag12', 'TS_diff1', 'TS_diff2', 'TS_diff3',
        'TS_ma3', 'TS_ma6', 'TS_ma12', 'TS_std3', 'TS_std6',
        'Tm_lag1', 'Tm_lag2', 'Tm_lag3', 'Tm_lag6', 'Tm_lag12', 'Tm_diff1', 'Tm_diff2', 'Tm_diff3',
        'Tm_ma3', 'Tm_ma6', 'Tm_ma12', 'Tm_std3', 'Tm_std6',
        'TS_Tm_ratio', 'TS_Tm_product', 'TS_Tm_diff',
        'PS_lag1', 'PS_lag2', 'PS_diff1', 'PS_diff2', 'PS_ma3', 'PS_ma6', 'PS_std3',
        'WPS_lag1', 'WPS_lag2', 'WPS_diff1', 'WPS_diff2', 'WPS_ma3', 'WPS_ma6', 'WPS_std3'
    ]
    feats = [f for f in feats if f in df.columns]
    target = ['TS']

    X, y, seasons, times, station_ids, years, doys = [], [], [], [], [], [], []

    group_cols = [c for c in ['station_id', 'ELV'] if c in df.columns]
    if group_cols:
        grouped = df.groupby(group_cols)
    else:
        grouped = [(None, df)]

    for key, g in grouped:
        if 'TIME' in g.columns:
            g = g.sort_values('TIME')
            time_vals = pd.to_datetime(g['TIME'], errors='coerce').values
        else:
            time_vals = np.arange(len(g))

        if 'station_id' in g.columns:
            current_station = str(g['station_id'].iloc[0])
        else:
            current_station = '-1'

        data_x = g[feats].values
        data_y = g[target].values
        season_vals = g['season'].values if 'season' in g.columns else None
        year_vals = g['YEAR'].values if 'YEAR' in g.columns else None
        doy_vals = g['DOY'].values if 'DOY' in g.columns else None

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
            if doy_vals is not None:
                doys.append(doy_vals[i + time_steps])

    X = np.array(X)
    y = np.array(y)
    seasons = np.array(seasons) if len(seasons) == len(X) and len(seasons) > 0 else None
    times = np.array(times)
    station_ids = np.array(station_ids)
    years = np.array(years) if 'YEAR' in df.columns else None
    doys = np.array(doys) if 'DOY' in df.columns else None

    if len(X) == 0 or len(y) == 0:
        raise ValueError(f"序列生成失败：有效序列数为0，time_steps={time_steps}")
    return X, y, seasons, times, station_ids, years, doys


def split_stations(station_ids, test_station_ratio=0.1, random_state=42):
    unique_stations = np.unique(station_ids)
    rng = np.random.RandomState(random_state)
    shuffled_stations = rng.permutation(unique_stations)
    n_test_stations = max(1, int(len(unique_stations) * test_station_ratio))
    test_stations = shuffled_stations[:n_test_stations]
    train_stations = shuffled_stations[n_test_stations:]
    return test_stations, train_stations


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
    x_aug = x.copy()

    if noise_level and noise_level > 0:
        noise = np.random.normal(0, noise_level, x.shape)
        x_aug = x_aug + noise

    if np.random.random() < 0.3:
        dropout_prob = 0.05
        mask = np.random.binomial(1, 1 - dropout_prob, size=x_aug.shape)
        x_aug = x_aug * mask

    return x_aug


class TimeSeriesDataset(Dataset):
    "PyTorch时序数据集，支持训练集数据增强（内存优化版）"

    def __init__(self, X, y, augment=False, noise_level=0.005):
        self.X = X
        self.y = y
        self.augment = augment
        self.noise_level = noise_level

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx].astype(np.float32)
        y = self.y[idx].astype(np.float32)

        if self.augment:
            x = augment_time_series(x, noise_level=self.noise_level)

        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


class GRUModel(nn.Module):
    "增强的GRU模型（无注意力机制，输出TS）"

    def __init__(self, input_size, hidden_size=128, num_layers=3, output_size=1, time_steps=24, dropout=0.2):
        super(GRUModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.time_steps = time_steps

        self.input_ln = nn.LayerNorm(input_size)

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        self.ln = nn.LayerNorm(hidden_size)

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
        batch_size, seq_len, input_dim = x.shape
        x = self.input_ln(x)

        gru_out, _ = self.gru(x)
        out = self.ln(gru_out[:, -1, :])
        out = self.fc_net(out)
        return out


def train_model(train_loader, val_loader, input_size, time_steps, model_type='lstm', output_size=1, epochs=100, **kwargs):
    learning_rate = kwargs.get('learning_rate', 0.0005)
    hidden_size = kwargs.get('hidden_size', 128)
    dropout = kwargs.get('dropout', 0.2)
    weight_decay = kwargs.get('weight_decay', 1e-4)
    num_layers = kwargs.get('num_layers', 3)

    if model_type == 'gru':
        model = GRUModel(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            output_size=output_size,
            time_steps=time_steps,
            dropout=dropout
        ).to(device)
        model_name = "GRU"
    else:
        raise ValueError(f"不支持的模型类型: {model_type}")

    loss_function = 'log_cosh'
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
    else:
        criterion = nn.MSELoss()

    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6
    )

    best_val_loss = float('inf')
    patience = 15
    patience_counter = 0
    best_model_state = None

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()
            train_loss += loss.item() * batch_X.size(0)
            del batch_X, batch_y, outputs, loss
        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item() * batch_X.size(0)
                del batch_X, batch_y, outputs, loss
        val_loss /= len(val_loader.dataset)

        scheduler.step()

        if (epoch + 1) % 5 == 0:
            print(
                f'Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, LR: {optimizer.param_groups[0]["lr"]:.6f}', flush=True)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'Early stopping at epoch {epoch + 1}', flush=True)
                break

        if (epoch + 1) % 10 == 0:
            torch.cuda.empty_cache()

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    return model


def optimize_model_params(X_train, y_train, X_val, y_val, input_size, time_steps, model_type, output_size, n_trials=20):
    """使用optuna搜索模型最佳超参数"""
    print(f"开始{model_type}模型超参数搜索，搜索轮数: {n_trials}", flush=True)

    def objective(trial):
        # 基础参数
        params = {
            'learning_rate': trial.suggest_float('learning_rate', 1e-6, 1e-2, log=True),
            'hidden_size': trial.suggest_int('hidden_size', 64, 256),
            'dropout': trial.suggest_float('dropout', 0.1, 0.5),
            'weight_decay': trial.suggest_float('weight_decay', 1e-7, 1e-4, log=True),
            'num_layers': trial.suggest_int('num_layers', 1, 4)
        }

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
                epochs=50,
                **params
            )

            # 评估模型
            model.eval()
            val_loss = 0.0
            loss_function = 'log_cosh'

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

    print("\n超参数搜索完成！")
    print(f"最佳验证损失: {study.best_value:.4f}")
    print("最佳超参数:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")

    return study.best_params


def predict_with_model(model, data_loader):
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


def save_predictions_by_year_and_station(output_dir, model_preds, y_true, station_ids, years, doys, target_names):
    assert len(model_preds) == len(y_true) == len(station_ids) == len(years) == len(doys), "输入长度不一致"

    for j, target_name in enumerate(target_names):
        results = {}

        for i in range(len(model_preds)):
            year = int(years[i])
            station_id = str(station_ids[i])
            doy = int(doys[i])
            pred = model_preds[i, j]
            true = y_true[i, j]

            error = pred - true

            if year not in results:
                results[year] = {}
            if station_id not in results[year]:
                results[year][station_id] = []

            results[year][station_id].append({
                'DOY': doy,
                'StationID': station_id,
                'Predict': float(pred),
                'True': float(true),
                'Error': float(error)
            })

        target_dir = os.path.join(output_dir, target_name)

        for year, stations in results.items():
            year_dir = os.path.join(target_dir, f"year_{year}")
            os.makedirs(year_dir, exist_ok=True)

            for station_id, data in stations.items():
                file_path = os.path.join(year_dir, f"{station_id}.txt")
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write("DOY StationID Predict True Error\n")
                    for item in data:
                        f.write(f"{item['DOY']} {item['StationID']} {item['Predict']:.6f} {item['True']:.6f} {item['Error']:.6f}\n")

        print(f"{target_name}预测结果已按年份和站点保存至 {target_dir}", flush=True)


def evaluate_models(y_test, model, test_loader, model_type="GRU"):
    model_preds, _ = predict_with_model(model, test_loader)

    model_metrics = {}
    target_names = ['TS']

    for j, target_name in enumerate(target_names):
        model_metrics[target_name] = {
            'mse': mean_squared_error(y_test[:, j], model_preds[:, j]),
            'rmse': np.sqrt(mean_squared_error(y_test[:, j], model_preds[:, j])),
            'mae': mean_absolute_error(y_test[:, j], model_preds[:, j]),
            'r2': r2_score(y_test[:, j], model_preds[:, j])
        }

    print(f"\n=== {model_type}模型评估（标准化尺度）===")
    for target_name in target_names:
        print(
            f"{target_name}: MSE: {model_metrics[target_name]['mse']:.4f}, RMSE: {model_metrics[target_name]['rmse']:.4f}, "
            f"MAE: {model_metrics[target_name]['mae']:.4f}, R2: {model_metrics[target_name]['r2']:.4f}")

    return {'model': model_metrics}, model_preds


def main():
    root_dir = r"/share/home/u23114/tj23114/packages/dachuang_pwv/PS/xg_data"
    output_dir = r"/share/home/u23114/tj23114/packages/dachuang_pwv/Ts_Tm/ts_result_a800"

    time_steps = 24
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

        print("移除异常值...", flush=True)
        outlier_cols = ['TS']
        result_data = remove_outliers(result_data, outlier_cols, method='iqr')

        numeric_cols = ['YEAR', 'DOY', 'LAT', 'LON', 'ELV', 'TS', 'PS', 'WPS', 'Tm',
                        'hour', 'month', 'day', 'weekday', 'season',
                        'lat_sin', 'lat_cos', 'lon_sin', 'lon_cos',
                        'TS_lag1', 'TS_lag2', 'TS_lag3', 'TS_lag6', 'TS_lag12', 'TS_diff1', 'TS_diff2', 'TS_diff3',
                        'TS_ma3', 'TS_ma6', 'TS_ma12', 'TS_std3', 'TS_std6',
                        'Tm_lag1', 'Tm_lag2', 'Tm_lag3', 'Tm_lag6', 'Tm_lag12', 'Tm_diff1', 'Tm_diff2', 'Tm_diff3',
                        'Tm_ma3', 'Tm_ma6', 'Tm_ma12', 'Tm_std3', 'Tm_std6',
                        'TS_Tm_ratio', 'TS_Tm_product', 'TS_Tm_diff',
                        'PS_ma3', 'PS_ma6', 'PS_std3',
                        'WPS_ma3', 'WPS_ma6', 'WPS_std3',
                        'elevation_normalized', 'distance_from_equator']
        present_cols = [c for c in numeric_cols if c in result_data.columns]
        result_data[present_cols] = result_data[present_cols].astype(np.float32)

        X_seq, y_seq, seasons, times, station_ids, years, doys = create_sequences(result_data, time_steps)
        feature_dim = X_seq.shape[2]
        target_names = ['TS']
        print(f"序列形状：X={X_seq.shape}, y={y_seq.shape}（目标：TS）", flush=True)

        print("按站点和年份划分数据集...", flush=True)
        if 'YEAR' in result_data.columns and years is not None and station_ids is not None:
            test_stations, train_stations = split_stations(station_ids, test_station_ratio=0.1, random_state=42)

            val_stations_mask = np.isin(station_ids, test_stations)
            X_val_stations = X_seq[val_stations_mask]
            y_val_stations = y_seq[val_stations_mask]
            doys_val_stations = doys[val_stations_mask] if doys is not None else None

            train_station_mask = np.isin(station_ids, train_stations)
            X_train_stations = X_seq[train_station_mask]
            y_train_stations = y_seq[train_station_mask]
            years_train_stations = years[train_station_mask]
            doys_train_stations = doys[train_station_mask] if doys is not None else None

            train_mask = (years_train_stations >= 2014) & (years_train_stations <= 2016)
            test_mask = (years_train_stations == 2017)
            val_mask = (years_train_stations == 2018)

            X_train = X_train_stations[train_mask]
            y_train = y_train_stations[train_mask]
            doys_train = doys_train_stations[train_mask] if doys_train_stations is not None else None
            X_test = X_train_stations[test_mask]
            y_test = y_train_stations[test_mask]
            doys_test = doys_train_stations[test_mask] if doys_train_stations is not None else None
            X_val_years = X_train_stations[val_mask]
            y_val_years = y_train_stations[val_mask]
            doys_val_years = doys_train_stations[val_mask] if doys_train_stations is not None else None

            X_val = np.vstack([X_val_years, X_val_stations]) if len(X_val_years) > 0 else X_val_stations
            y_val = np.vstack([y_val_years, y_val_stations]) if len(y_val_years) > 0 else y_val_stations
            doys_val = np.hstack([doys_val_years, doys_val_stations]) if doys_val_years is not None and len(doys_val_years) > 0 else doys_val_stations

            print(f"按站点和年份拆分：")
            print(f"训练集（90%站点，14-16年）: {len(X_train)} 样本")
            print(f"测试集（90%站点，17年）: {len(X_test)} 样本")
            print(f"验证集（90%站点，18年 + 10%站点所有年份）: {len(X_val)} 样本")
            print(f"10%验证站点数量: {len(test_stations)}")
            print(f"90%训练测试站点数量: {len(train_stations)}")
        else:
            raise ValueError("需要YEAR列和station_ids数据")

        print("特征标准化...", flush=True)
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()

        n_samples, n_steps, n_features = X_train.shape
        X_train_flat = X_train.reshape(-1, n_features)
        scaler_X.fit(X_train_flat)
        X_train = scaler_X.transform(X_train_flat).reshape(n_samples, n_steps, n_features)

        n_val_samples = X_val.shape[0]
        X_val_flat = X_val.reshape(-1, n_features)
        X_val = scaler_X.transform(X_val_flat).reshape(n_val_samples, n_steps, n_features)

        n_test_samples = X_test.shape[0]
        X_test_flat = X_test.reshape(-1, n_features)
        X_test = scaler_X.transform(X_test_flat).reshape(n_test_samples, n_steps, n_features)

        y_train = scaler_y.fit_transform(y_train)
        y_val = scaler_y.transform(y_val)
        y_test = scaler_y.transform(y_test)

        print("\n开始超参数搜索...", flush=True)
        best_params = optimize_model_params(
            X_train, y_train, X_val, y_val, feature_dim, time_steps, model_type='gru', output_size=1, n_trials=20
        )

        print("\n使用最佳超参数训练最终模型...", flush=True)
        batch_size = best_params.pop('batch_size', 32)
        train_dataset = TimeSeriesDataset(X_train, y_train, augment=True)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_dataset = TimeSeriesDataset(X_val, y_val, augment=False)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        test_dataset = TimeSeriesDataset(X_test, y_test, augment=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)

        final_model = train_model(
            train_loader, val_loader,
            input_size=feature_dim,
            time_steps=time_steps,
            model_type='gru',
            output_size=1,
            epochs=200,
            **best_params
        )

        metrics, model_preds = evaluate_models(y_test, final_model, test_loader, model_type="GRU")

        model_preds_original = scaler_y.inverse_transform(model_preds)
        y_test_original = scaler_y.inverse_transform(y_test)

        scatter_dir = os.path.join(output_dir, "gru")
        os.makedirs(scatter_dir, exist_ok=True)

        plot_scatter(y_test_original, model_preds_original, 
                    os.path.join(scatter_dir, "scatter_gru_test_TS.png"), 
                    title="GRU Test Set (TS)")

        model_path = os.path.join(scatter_dir, "gru_model.pth")
        torch.save({
            'model_state_dict': final_model.state_dict(),
            'scaler_X': scaler_X,
            'scaler_y': scaler_y,
            'input_size': feature_dim,
            'time_steps': time_steps,
            'best_params': best_params
        }, model_path)
        print(f"模型已保存至 {model_path}", flush=True)

        save_predictions_by_year_and_station(
            os.path.join(scatter_dir, "predictions"),
            model_preds_original,
            y_test_original,
            station_ids[train_station_mask][test_mask] if 'train_station_mask' in locals() and 'test_mask' in locals() else station_ids,
            years[train_station_mask][test_mask] if 'train_station_mask' in locals() and 'test_mask' in locals() else years,
            doys_test,
            target_names
        )

    except Exception as e:
        print(f"运行出错: {e}", flush=True)
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
