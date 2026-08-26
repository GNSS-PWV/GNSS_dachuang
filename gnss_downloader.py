"""
GNSS ZTD数据下载与处理模块
支持多个数据源:
- IGS武汉大学对流层产品
- 美国SuomiNet对流层产品 (EarthScope)
- 澳大利亚对流层产品
- 欧洲对流层产品 (EUREF)
- UCAR COSMIC PWV产品
注: Nevada Geodetic Laboratory (NGL) 已移除，使用IGS全球网络作为站点列表来源
"""

import os
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import yaml
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import gzip
import xarray as xr
from scipy.interpolate import RegularGridInterpolator
from scipy.spatial import cKDTree
from functools import lru_cache
import time
import logging
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    print("提示: 安装tqdm可获得进度条显示 (pip install tqdm)")


# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DownloadStats:
    """下载统计类，用于跟踪下载进度和成功率"""

    def __init__(self):
        self.total_attempts = 0
        self.successful_downloads = 0
        self.failed_downloads = 0
        self.cached_files = 0
        self.total_bytes = 0
        self.start_time = None
        self.end_time = None
        self.errors = []

    def start(self):
        """开始统计"""
        self.start_time = time.time()

    def end(self):
        """结束统计"""
        self.end_time = time.time()

    def record_success(self, file_size_bytes: int = 0):
        """记录成功下载"""
        self.total_attempts += 1
        self.successful_downloads += 1
        self.total_bytes += file_size_bytes

    def record_failure(self, error_msg: str = ""):
        """记录失败下载"""
        self.total_attempts += 1
        self.failed_downloads += 1
        if error_msg:
            self.errors.append(error_msg)

    def record_cached(self):
        """记录缓存命中"""
        self.total_attempts += 1
        self.cached_files += 1

    def get_summary(self) -> Dict[str, any]:
        """获取统计摘要"""
        duration = (self.end_time - self.start_time) if self.end_time and self.start_time else 0
        success_rate = (self.successful_downloads / self.total_attempts * 100) if self.total_attempts > 0 else 0
        avg_speed = (self.total_bytes / duration / 1024) if duration > 0 else 0  # KB/s

        return {
            'total_attempts': self.total_attempts,
            'successful': self.successful_downloads,
            'failed': self.failed_downloads,
            'cached': self.cached_files,
            'success_rate': success_rate,
            'total_size_mb': self.total_bytes / (1024 * 1024),
            'duration_seconds': duration,
            'avg_speed_kbps': avg_speed,
            'error_count': len(self.errors)
        }

    def print_summary(self):
        """打印统计摘要"""
        summary = self.get_summary()

        print("\n" + "=" * 60)
        print("下载统计报告")
        print("=" * 60)
        print(f"总尝试次数:     {summary['total_attempts']}")
        print(f"成功下载:       {summary['successful']} ({summary['success_rate']:.1f}%)")
        print(f"失败下载:       {summary['failed']}")
        print(f"缓存命中:       {summary['cached']}")
        print(f"总下载量:       {summary['total_size_mb']:.2f} MB")
        print(f"耗时:           {summary['duration_seconds']:.1f} 秒")
        print(f"平均速度:       {summary['avg_speed_kbps']:.1f} KB/s")

        if self.errors:
            print(f"\n错误数量:       {summary['error_count']}")
            print("最近错误:")
            for error in self.errors[-5:]:  # 显示最近5个错误
                print(f"  - {error}")

        print("=" * 60)


def retry_on_failure(max_retries=3, delay=1.0, backoff=2.0):
    """
    重试装饰器，用于网络请求失败时自动重试

    Args:
        max_retries: 最大重试次数
        delay: 初始延迟时间（秒）
        backoff: 延迟时间的倍增因子
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            current_delay = delay
            last_exception = None

            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        logger.warning(
                            f"{func.__name__} 失败 (尝试 {attempt + 1}/{max_retries}): {e}"
                        )
                        time.sleep(current_delay)
                        current_delay *= backoff
                    else:
                        logger.error(
                            f"{func.__name__} 最终失败 (尝试 {max_retries} 次): {e}"
                        )

            raise last_exception

        return wrapper
    return decorator


class ERA5MeteoProvider:
    """ERA5气象数据提供器，用于PWV计算（优化版：使用LRU缓存）"""

    def __init__(self, era5_dir: str = "data/raw/labels/ERA5", cache_size: int = 10):
        """
        初始化ERA5气象数据提供器

        Args:
            era5_dir: ERA5数据目录
            cache_size: LRU缓存大小（最多缓存多少天的数据）
        """
        self.era5_dir = Path(era5_dir)
        self.cache_size = cache_size
        self._cache = {}  # 基础缓存
        self._access_order = []  # LRU访问顺序

    def _load_era5_file(self, date_str: str) -> Optional[xr.Dataset]:
        """
        加载ERA5文件（带LRU缓存）

        Args:
            date_str: 日期字符串 YYYYMMDD

        Returns:
            xarray Dataset或None
        """
        # 检查缓存
        if date_str in self._cache:
            # 更新访问顺序（移到最后）
            self._access_order.remove(date_str)
            self._access_order.append(date_str)
            return self._cache[date_str]

        # 加载文件
        era5_file = self.era5_dir / f"ERA5_Multi_{date_str}.nc"

        if not era5_file.exists():
            logger.warning(f"ERA5文件不存在: {era5_file}")
            return None

        try:
            ds = xr.open_dataset(era5_file)

            # 添加到缓存
            self._cache[date_str] = ds
            self._access_order.append(date_str)

            # LRU淘汰：如果缓存超过限制，移除最久未使用的
            if len(self._cache) > self.cache_size:
                oldest_key = self._access_order.pop(0)
                old_ds = self._cache.pop(oldest_key)
                old_ds.close()
                logger.debug(f"LRU淘汰: {oldest_key}")

            return ds

        except Exception as e:
            logger.error(f"读取ERA5文件失败 {era5_file}: {e}")
            return None

    def get_meteo_data(
        self,
        lat: float,
        lon: float,
        time: datetime
    ) -> Dict[str, float]:
        """
        获取指定位置和时间的气象数据（优化版：使用LRU缓存）

        Args:
            lat: 纬度
            lon: 经度
            time: 时间

        Returns:
            包含temperature_k和pressure_pa的字典
        """
        # 查找对应日期的ERA5文件
        date_str = time.strftime('%Y%m%d')

        # 使用LRU缓存加载
        ds = self._load_era5_file(date_str)

        if ds is None:
            # 如果文件不存在，返回默认值
            logger.warning(f"使用默认气象值 (日期: {date_str})")
            return {
                'temperature_k': 288.0,  # 默认288K
                'pressure_pa': 101325.0   # 默认1013.25 hPa
            }

        # 找到最接近的时间步
        try:
            time_diff = abs(ds.time - np.datetime64(time))
            closest_time_idx = int(time_diff.argmin())

            # 获取温度和压强数据
            temp_data = ds['t2m'].isel(time=closest_time_idx)  # K
            pres_data = ds['sp'].isel(time=closest_time_idx)   # Pa

            # 空间插值到指定位置
            temperature_k = float(temp_data.interp(
                latitude=lat,
                longitude=lon,
                method='linear'
            ).values)

            pressure_pa = float(pres_data.interp(
                latitude=lat,
                longitude=lon,
                method='linear'
            ).values)

            return {
                'temperature_k': temperature_k,
                'pressure_pa': pressure_pa
            }

        except Exception as e:
            logger.warning(f"插值ERA5数据失败: {e}")
            return {
                'temperature_k': 288.0,
                'pressure_pa': 101325.0
            }

    def get_cache_stats(self) -> Dict[str, int]:
        """获取缓存统计信息"""
        return {
            'cached_days': len(self._cache),
            'cache_size_limit': self.cache_size,
            'cache_usage_pct': len(self._cache) / self.cache_size * 100
        }

    def close(self):
        """关闭所有缓存的数据集"""
        for ds in self._cache.values():
            ds.close()
        self._cache.clear()
        self._access_order.clear()
        logger.info("ERA5缓存已清理")


class EarthScopeAuth:
    """EarthScope认证管理器"""

    def __init__(self, token_file: str = None):
        """
        初始化EarthScope认证

        Args:
            token_file: token文件路径，如果为None则自动搜索
        """
        # 尝试多个可能的token文件路径
        possible_paths = [
            "download_lables/access_token_new.txt",
            "download_lables/download_lables/access_token_new.txt",
            "download_lables/access_token.txt",
        ]

        if token_file:
            possible_paths.insert(0, token_file)

        self.token_file = None
        for path in possible_paths:
            p = Path(path)
            if p.exists():
                self.token_file = p
                break

        self.access_token = None
        self.refresh_token = None
        self.expires_at = None

        # 尝试从文件读取token
        if self.token_file and self.token_file.exists():
            self._load_token()

    def _load_token(self):
        """从文件加载token"""
        try:
            with open(self.token_file, 'r') as f:
                token_data = json.load(f)
                self.access_token = token_data.get('access_token')
                self.refresh_token = token_data.get('refresh_token')
                self.expires_at = token_data.get('expires_at', 0)
                print("EarthScope token已加载")
        except Exception as e:
            print(f"警告: 读取token文件失败: {e}")

    def _save_token(self):
        """将当前token保存到文件"""
        if not self.token_file:
            return
        try:
            token_data = {
                'access_token': self.access_token,
                'refresh_token': self.refresh_token,
                'expires_at': int(self.expires_at) if self.expires_at else 0,
                'issued_at': int(datetime.now().timestamp()),
                'scope': 'openid profile email offline_access'
            }
            with open(self.token_file, 'w') as f:
                json.dump(token_data, f, indent=2)
            print(f"EarthScope token已保存到 {self.token_file}")
        except Exception as e:
            print(f"警告: 保存token失败: {e}")

    def _refresh_access_token(self) -> bool:
        """使用refresh_token刷新access_token (通过requests直接调用OAuth2端点)"""
        if not self.refresh_token:
            return False
        try:
            r = requests.post(
                'https://login.earthscope.org/oauth/token',
                headers={'content-type': 'application/x-www-form-urlencoded'},
                data={
                    'grant_type': 'refresh_token',
                    'client_id': 'b9DtAFBd6QvMg761vI3YhYquNZbJX5G0',
                    'refresh_token': self.refresh_token,
                    'scope': 'openid profile email offline_access',
                },
                timeout=30
            )
            if r.status_code != 200:
                print(f"警告: token刷新请求失败, status={r.status_code}")
                return False
            new_tokens = r.json()
            self.access_token = new_tokens['access_token']
            self.expires_at = int(datetime.now().timestamp() + new_tokens.get('expires_in', 28800))
            if 'refresh_token' in new_tokens:
                self.refresh_token = new_tokens['refresh_token']
            self._save_token()
            print(f"EarthScope token已自动刷新, 有效期至 {datetime.fromtimestamp(self.expires_at)}")
            return True
        except Exception as e:
            print(f"警告: 自动刷新token失败: {e}")
            return False

    def get_access_token(self) -> str:
        """
        获取访问token（如果过期则自动刷新）

        Returns:
            访问token
        """
        # token未过期，直接返回
        if self.expires_at and datetime.now().timestamp() < self.expires_at - 300:
            return self.access_token

        # token过期，尝试用refresh_token自动刷新
        print("EarthScope token已过期，尝试自动刷新...")
        if self._refresh_access_token():
            return self.access_token

        # 自动刷新失败，尝试重新加载文件（可能被外部脚本更新）
        if self.token_file and self.token_file.exists():
            self._load_token()
            if self.expires_at and datetime.now().timestamp() < self.expires_at - 300:
                return self.access_token

        # 所有刷新手段失败
        print("警告: EarthScope token已过期且无法自动刷新")
        print("请运行设备代码流认证: python download_lables/earthscope_auth.py")

        if self.access_token:
            return self.access_token

        raise ValueError("EarthScope token不可用，请先运行认证流程")


class GNSSDownloader:
    """
    GNSS数据下载器 - 支持多个数据源

    数据源:
    1. IGS武汉大学对流层产品 (全球骨干网络，ZTD)
    2. SuomiNet (EarthScope) (美国高密度网络，ZTD，需认证)
    3. 澳大利亚对流层产品 (澳洲+亚太，ZTD)
    4. 欧洲对流层产品 (EUREF) (欧洲，ZTD)
    5. UCAR COSMIC PWV产品 (美国CONUS，PWV直接产品)
    """

    # 数据源URL配置
    DATA_SOURCES = {
        'igs_whu': {
            'url': "ftp://igs.gnsswhu.cn/pub/gps/products/troposphere/new",
            'description': "IGS武汉大学对流层产品 - 全球骨干网络，ZTD产品，无需认证",
            'coverage': "全球",
            'product': "ZTD",
            'auth_required': False,
            'station_file': "IGSwhu_formatted.txt"
        },
        'suominet': {
            'url': "https://gage-data.earthscope.org/archive/gnss/products/troposphere",
            'description': "EarthScope SuomiNet - 美国高密度网络，ZTD产品，需OAuth2认证",
            'coverage': "美国",
            'product': "ZTD",
            'auth_required': True,
            'station_file': "suominet_formatted.txt"
        },
        'australia': {
            'url': "https://ga-gnss-products-v1.s3.amazonaws.com",
            'description': "澳大利亚气象局 - 澳洲+亚太区域，ZTD产品，无需认证",
            'coverage': "澳洲+亚太",
            'product': "ZTD",
            'auth_required': False,
            'station_file': "austrilian_stations_formatted.txt"
        },
        'euref': {
            'url': "https://igs.bkg.bund.de/root_ftp/EUREF/products",
            'description': "EUREF - 欧洲网络，ZTD产品，无需认证",
            'coverage': "欧洲",
            'product': "ZTD",
            'auth_required': False,
            'station_file': "euref_formatted.txt"
        },
        'ucar_cosmic': {
            'url': "https://data.cosmic.ucar.edu/suominet/postProcess/pwvConus",
            'description': "UCAR COSMIC - 美国CONUS区域，PWV直接产品，无需认证",
            'coverage': "美国CONUS",
            'product': "PWV",
            'auth_required': False,
            'station_file': "ucar_cosmic_formatted.txt"
        }
    }

    def __init__(
        self,
        config_path: str = "config_v2.yaml",
        data_source: str = "igs_whu",
        era5_dir: str = "data/raw/labels/ERA5",
        station_list_dir: str = "download_lables/GNSS_list"
    ):
        """
        初始化GNSS下载器

        Args:
            config_path: 配置文件路径
            data_source: 数据源 ('igs_whu', 'suominet', 'australia', 'euref', 'ucar_cosmic')
            era5_dir: ERA5数据目录（用于PWV计算）
            station_list_dir: 本地站点列表目录
        """
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)

        self.data_source = data_source
        self.save_dir = Path("data/raw/validation/GNSS")
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # 本地站点列表目录
        self.station_list_dir = Path(station_list_dir)

        # 站点列表文件
        self.stations_file = self.save_dir / "igs_stations.csv"

        # ERA5气象数据提供器
        self.era5_provider = ERA5MeteoProvider(era5_dir)

        # EarthScope认证（用于SuomiNet）
        self.earthscope_auth = None
        if data_source == 'suominet':
            self.earthscope_auth = EarthScopeAuth()

        # 下载统计
        self.stats = DownloadStats()

    def load_station_list_from_file(
        self,
        source: Optional[str] = None
    ) -> Optional[pd.DataFrame]:
        """
        从本地文件加载站点列表

        格式: SITE_ID, LAT, LON, HEIGHT
        例如: ABMF00GLP, 16.262, -61.528, -25.268

        Args:
            source: 数据源名称，如果为None则使用当前数据源

        Returns:
            站点列表DataFrame，如果文件不存在则返回None
        """
        if source is None:
            source = self.data_source

        # 获取数据源对应的站点列表文件
        if source not in self.DATA_SOURCES:
            print(f"警告: 未知数据源 {source}")
            return None

        station_file_name = self.DATA_SOURCES[source].get('station_file')
        if not station_file_name:
            print(f"警告: 数据源 {source} 未配置站点列表文件")
            return None

        station_file_path = self.station_list_dir / station_file_name

        if not station_file_path.exists():
            print(f"警告: 站点列表文件不存在: {station_file_path}")
            return None

        try:
            print(f"正在加载本地站点列表: {station_file_path}")

            # 读取文件（格式: SITE_ID, LAT, LON, HEIGHT）
            data = []
            with open(station_file_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue

                    parts = line.split(',')
                    if len(parts) >= 3:
                        try:
                            station_id = parts[0].strip()
                            lat = float(parts[1].strip())
                            lon = float(parts[2].strip())
                            height = float(parts[3].strip()) if len(parts) >= 4 else 0.0

                            data.append({
                                'station_id': station_id,
                                'lat': lat,
                                'lon': lon,
                                'height': height
                            })
                        except (ValueError, IndexError) as e:
                            continue

            if data:
                df = pd.DataFrame(data)
                print(f"成功加载 {len(df)} 个站点 (来源: {station_file_name})")

                # 保存到缓存文件
                cache_file = self.save_dir / f"{source}_stations.csv"
                df.to_csv(cache_file, index=False)

                return df
            else:
                print(f"警告: 文件中没有有效的站点数据")
                return None

        except Exception as e:
            print(f"错误: 加载站点列表失败 - {e}")
            return None

    def download_station_list(self) -> pd.DataFrame:
        """
        下载全球GNSS站点列表（优先从本地加载，失败后从IGS网络下载）

        Returns:
            站点信息DataFrame
        """
        # 1. 优先尝试从本地文件加载
        df_local = self.load_station_list_from_file()
        if df_local is not None and not df_local.empty:
            return df_local

        # 2. 如果本地加载失败，尝试从在线下载
        print("本地站点列表不可用，尝试从IGS在线下载...")
        print("正在下载IGS全球站点列表...")

        # 尝试多个IGS数据源
        urls = [
            # IGS站点列表（推荐）
            "https://files.igs.org/pub/station/general/IGSNetwork.csv",
            # IGS备用源
            "http://www.igs.org/network/netindex/IGSNetwork.csv",
        ]

        for url in urls:
            try:
                print(f"尝试: {url}")
                response = requests.get(url, timeout=30)
                response.raise_for_status()

                # IGS格式通常是CSV
                from io import StringIO
                df = pd.read_csv(StringIO(response.text))

                # 标准化列名（IGS格式可能不同）
                # 通常包含: Station, Latitude, Longitude, Height等
                if 'Station' in df.columns:
                    df = df.rename(columns={
                        'Station': 'station_id',
                        'Latitude': 'lat',
                        'Longitude': 'lon',
                        'Height': 'height'
                    })

                # 确保必要列存在
                required_cols = ['station_id', 'lat', 'lon']
                if not all(col in df.columns for col in required_cols):
                    print(f"  格式不匹配，尝试下一个源...")
                    continue

                # 添加默认高度如果缺失
                if 'height' not in df.columns:
                    df['height'] = 0.0

                # 转换站点ID为小写
                df['station_id'] = df['station_id'].str.lower()

                # 保存
                df.to_csv(self.stations_file, index=False)
                print(f"站点列表已保存: {self.stations_file}, 总计 {len(df)} 个站点")
                return df

            except Exception as e:
                print(f"  失败: {e}")
                continue

        # 如果所有URL都失败，使用fallback：创建基本的全球站点列表
        print("\n所有IGS源均失败，使用fallback全球站点列表...")
        fallback_stations = self._create_fallback_station_list()
        fallback_stations.to_csv(self.stations_file, index=False)
        print(f"Fallback站点列表已保存: {self.stations_file}, 总计 {len(fallback_stations)} 个站点")
        return fallback_stations

    def _create_fallback_station_list(self) -> pd.DataFrame:
        """
        创建fallback全球GNSS站点列表
        包含主要IGS核心站点，分布在全球各大洲

        Returns:
            基本站点列表DataFrame
        """
        # 主要IGS核心站点（全球分布）
        fallback_stations = [
            # 亚洲
            {'station_id': 'bjfs', 'lat': 39.6086, 'lon': 115.8925, 'height': 88.5, 'region': 'China'},
            {'station_id': 'urum', 'lat': 43.8081, 'lon': 87.6006, 'height': 858.4, 'region': 'China'},
            {'station_id': 'lhaz', 'lat': 29.6572, 'lon': 91.1040, 'height': 3624.6, 'region': 'China'},
            {'station_id': 'tskb', 'lat': 36.1056, 'lon': 140.0869, 'height': 76.5, 'region': 'Japan'},
            {'station_id': 'usud', 'lat': 36.1331, 'lon': 138.3628, 'height': 1456.4, 'region': 'Japan'},
            {'station_id': 'iisc', 'lat': 13.0211, 'lon': 77.5706, 'height': 844.0, 'region': 'India'},
            {'station_id': 'darw', 'lat': -12.8438, 'lon': 131.1327, 'height': 124.6, 'region': 'Australia'},
            {'station_id': 'alic', 'lat': -23.6701, 'lon': 133.8855, 'height': 603.3, 'region': 'Australia'},
            {'station_id': 'pert', 'lat': -31.8020, 'lon': 115.8853, 'height': 31.3, 'region': 'Australia'},

            # 欧洲
            {'station_id': 'onsa', 'lat': 57.3958, 'lon': 11.9256, 'height': 45.0, 'region': 'Sweden'},
            {'station_id': 'pots', 'lat': 52.3794, 'lon': 13.0661, 'height': 144.3, 'region': 'Germany'},
            {'station_id': 'zimm', 'lat': 46.8771, 'lon': 7.4652, 'height': 956.4, 'region': 'Switzerland'},
            {'station_id': 'brux', 'lat': 50.7978, 'lon': 4.3592, 'height': 156.7, 'region': 'Belgium'},
            {'station_id': 'graz', 'lat': 47.0671, 'lon': 15.4935, 'height': 538.4, 'region': 'Austria'},
            {'station_id': 'mate', 'lat': 40.6492, 'lon': 16.7044, 'height': 535.8, 'region': 'Italy'},

            # 北美
            {'station_id': 'algo', 'lat': 45.9558, 'lon': -78.0714, 'height': 201.4, 'region': 'Canada'},
            {'station_id': 'yell', 'lat': 62.4809, 'lon': -114.4810, 'height': 180.6, 'region': 'Canada'},
            {'station_id': 'nrc1', 'lat': 45.4531, 'lon': -75.6184, 'height': 90.7, 'region': 'Canada'},
            {'station_id': 'mdop', 'lat': 30.6806, 'lon': -104.0150, 'height': 2004.4, 'region': 'USA'},
            {'station_id': 'gode', 'lat': 39.0217, 'lon': -76.8267, 'height': 27.5, 'region': 'USA'},
            {'station_id': 'pie1', 'lat': 34.3016, 'lon': -108.1191, 'height': 2364.9, 'region': 'USA'},

            # 南美
            {'station_id': 'braz', 'lat': -15.9475, 'lon': -47.8781, 'height': 1106.0, 'region': 'Brazil'},
            {'station_id': 'bogt', 'lat': 4.6401, 'lon': -74.0809, 'height': 2576.5, 'region': 'Colombia'},
            {'station_id': 'riog', 'lat': -53.7856, 'lon': -67.7512, 'height': 24.0, 'region': 'Argentina'},

            # 非洲
            {'station_id': 'hrao', 'lat': -25.8897, 'lon': 27.6869, 'height': 1414.8, 'region': 'South Africa'},
            {'station_id': 'har2', 'lat': -25.8871, 'lon': 27.7073, 'height': 1415.9, 'region': 'South Africa'},

            # 南极
            {'station_id': 'ohig', 'lat': -63.3211, 'lon': -57.9014, 'height': 34.6, 'region': 'Antarctica'},
            {'station_id': 'mcm4', 'lat': -77.8380, 'lon': 166.6693, 'height': 98.1, 'region': 'Antarctica'},

            # 太平洋岛屿
            {'station_id': 'guam', 'lat': 13.5893, 'lon': 144.8684, 'height': 188.2, 'region': 'Guam'},
            {'station_id': 'thti', 'lat': -17.5769, 'lon': -149.6061, 'height': 96.5, 'region': 'Tahiti'},
        ]

        df = pd.DataFrame(fallback_stations)
        print(f"创建了 {len(df)} 个fallback站点（全球分布）")
        return df

    def select_global_uniform_stations(
        self,
        n_stations: int = 1500,
        grid_size: float = 5.0
    ) -> pd.DataFrame:
        """
        全球均匀采样站点(避免欧美主导)

        Args:
            n_stations: 目标站点数
            grid_size: 网格大小(度), 用于均匀采样

        Returns:
            选中的站点DataFrame
        """
        # 读取或下载站点列表
        if not self.stations_file.exists():
            df_all = self.download_station_list()
        else:
            df_all = pd.read_csv(self.stations_file)

        print(f"正在从 {len(df_all)} 个站点中均匀采样 {n_stations} 个...")

        # 为每个站点分配网格ID
        df_all['grid_lat'] = (df_all['lat'] // grid_size).astype(int)
        df_all['grid_lon'] = (df_all['lon'] // grid_size).astype(int)
        df_all['grid_id'] = df_all['grid_lat'] * 360 + df_all['grid_lon']

        # 每个网格采样一个站点
        selected = []
        for grid_id, group in df_all.groupby('grid_id'):
            # 从每个网格中随机选择一个站点
            if len(selected) < n_stations:
                station = group.sample(n=1).iloc[0]
                selected.append(station)

        df_selected = pd.DataFrame(selected)

        # 如果数量不足, 从剩余站点中随机补充
        if len(df_selected) < n_stations:
            remaining = df_all[~df_all['station_id'].isin(df_selected['station_id'])]
            additional = remaining.sample(n=n_stations - len(df_selected))
            df_selected = pd.concat([df_selected, additional])

        print(f"采样完成: {len(df_selected)} 个站点")

        # 保存选中的站点
        selected_file = self.save_dir / "selected_stations_1500.csv"
        df_selected.to_csv(selected_file, index=False)

        return df_selected

    def download_suominet_ztd(
        self,
        date: datetime,
        patterns: List[str] = ['b', 'c']
    ) -> List[str]:
        """
        下载SuomiNet对流层产品数据

        文件命名规则: cwu{gps_week}{day_of_week}.{YYYYMMDD}.{pattern}.met.gz
        例如: cwu21960.20231101.b.met.gz

        Args:
            date: 日期
            patterns: 文件模式列表 (例如: ['b', 'c'])

        Returns:
            下载的文件路径列表
        """
        if self.earthscope_auth is None:
            raise ValueError("SuomiNet数据源需要EarthScope认证")

        # 获取token
        token = self.earthscope_auth.get_access_token()

        # 计算GPS周和星期几
        gps_epoch = datetime(1980, 1, 6)
        days_since_epoch = (date - gps_epoch).days
        gps_week = days_since_epoch // 7
        day_of_week = days_since_epoch % 7
        day_of_year = date.strftime("%j")
        date_str = date.strftime("%Y%m%d")
        year = date.year

        # 构建URL和保存路径
        base_url = f"{self.DATA_SOURCES['suominet']['url']}/{year}/{day_of_year}"
        downloaded_files = []

        for pattern in patterns:
            file_name = f"cwu{gps_week}{day_of_week}.{date_str}.{pattern}.met.gz"
            url = f"{base_url}/{file_name}"

            # 保存路径
            save_path = self.save_dir / "SuomiNet" / str(year) / day_of_year / file_name
            save_path.parent.mkdir(parents=True, exist_ok=True)

            # 检查文件是否已存在
            if save_path.exists():
                print(f"文件已存在: {save_path}")
                downloaded_files.append(str(save_path))
                continue

            try:
                # 下载文件
                print(f"正在下载SuomiNet数据: {file_name}")
                headers = {"authorization": f"Bearer {token}"}
                response = requests.get(url, headers=headers, timeout=60)
                response.raise_for_status()

                # 保存文件
                with open(save_path, 'wb') as f:
                    f.write(response.content)

                print(f"下载成功: {save_path}")
                downloaded_files.append(str(save_path))

            except requests.exceptions.RequestException as e:
                print(f"下载失败 {file_name}: {e}")

        return downloaded_files

    def parse_suominet_file(self, met_file: str) -> pd.DataFrame:
        """
        解析SuomiNet .met.gz文件

        文件格式 (新版):
        # SECS[J2000]  ZD[m] TROP_DRY[m] WETZ[m] WETZ_SIG[m] ... SITE MAP_FUNC
        752068800 2.315363 2.260854 0.054510 0.003601 ... AB09 VMF1

        SECS[J2000]: J2000纪元以来的秒数 (J2000 = 2000-01-01 12:00:00 UTC)
        ZD[m]: 天顶延迟，单位米
        WETZ_SIG[m]: 湿延迟的标准差，单位米

        Args:
            met_file: .met.gz文件路径

        Returns:
            DataFrame包含站点、时间和ZTD
        """
        data = []

        # J2000 epoch: 2000-01-01 12:00:00 UTC
        j2000_epoch = datetime(2000, 1, 1, 12, 0, 0)

        try:
            with gzip.open(met_file, 'rt') as f:
                for line in f:
                    if line.startswith('#') or not line.strip():
                        continue

                    parts = line.split()
                    if len(parts) >= 10:
                        # 解析新格式
                        # SECS[J2000]  ZD[m] TROP_DRY[m] WETZ[m] WETZ_SIG[m] ... SITE MAP_FUNC
                        try:
                            secs_j2000 = float(parts[0])  # J2000以来的秒数
                            zd_m = float(parts[1])  # 天顶延迟（米）
                            wetz_sig_m = float(parts[4])  # 湿延迟标准差（米）
                            site = parts[-2]  # 倒数第二列是站点ID

                            # 计算时间
                            dt = j2000_epoch + timedelta(seconds=secs_j2000)

                            # 转换为毫米
                            ztd_mm = zd_m * 1000
                            sigma_mm = wetz_sig_m * 1000

                            data.append({
                                'station_id': site,
                                'datetime': dt,
                                'ztd_mm': ztd_mm,
                                'sigma_mm': sigma_mm
                            })

                        except (ValueError, IndexError) as e:
                            # 跳过无法解析的行
                            continue

        except Exception as e:
            print(f"解析SuomiNet文件失败 {met_file}: {e}")

        return pd.DataFrame(data)

    def download_igs_whu_ztd(
        self,
        station_id: str,
        date: datetime,
        country_code: str = "CHN",
        rename_to_short: bool = True
    ) -> Optional[str]:
        """
        下载IGS武汉大学对流层产品（新格式）

        新格式URL: ftp://igs.gnsswhu.cn/pub/gps/products/troposphere/new/YYYY/DOY/
        文件名: IGS0OPSFIN_YYYYDOY0000_01D_05M_SSSSCCNNN_TRO.TRO.gz
        例如: IGS0OPSFIN_20230540000_01D_05M_BJFS00CHN_TRO.TRO.gz

        Args:
            station_id: 站点ID (4字符，例如: "bjfs")
            date: 日期
            country_code: 国家代码 (3字符，例如: "CHN", "USA")
            rename_to_short: 是否重命���为短格式 (例如: bjfs0540.23.tro.gz)

        Returns:
            下载文件路径
        """
        from ftplib import FTP
        import re

        year = date.year
        doy = date.strftime("%j")  # 一年中的天数
        yy = date.strftime("%y")   # 两位年份

        # 新格式文件名模式
        # IGS0OPSFIN_20230540000_01D_05M_BJFS00CHN_TRO.TRO.gz
        station_upper = station_id.upper()

        # 判断站点ID格式：长格式(9字符如BJFS00CHN)或短格式(4字符如BJFS)
        if len(station_upper) >= 9:
            # 长格式：直接使用完整站点ID
            file_pattern = f"IGS0OPSFIN_{year}{doy}0000_01D_05M_{station_upper}_TRO\\.TRO\\.gz"
        else:
            # 短格式：需要匹配后缀
            file_pattern = f"IGS0OPSFIN_{year}{doy}0000_01D_05M_{station_upper}\\d{{2}}\\w{{3}}_TRO\\.TRO\\.gz"

        # FTP路径
        ftp_host = "igs.gnsswhu.cn"
        ftp_path = f"/pub/gps/products/troposphere/new/{year}/{doy}"

        # 保存路径
        save_dir = self.save_dir / "IGS_WHU" / str(year) / doy
        save_dir.mkdir(parents=True, exist_ok=True)

        try:
            print(f"正在下载IGS武汉数据: {station_id.upper()}")

            # 连接FTP服务器
            ftp = FTP(ftp_host, timeout=30)
            ftp.login()  # 匿名登录

            # 切换到目标目录
            ftp.cwd(ftp_path)

            # 列出目录中的文件，找到匹配的文件
            files = ftp.nlst()

            # 查找匹配的文件
            matching_files = [f for f in files if re.match(file_pattern, f)]

            if not matching_files:
                print(f"未找到站点 {station_id.upper()} 的数据文件")
                ftp.quit()
                return None

            # 使用第一个匹配的文件
            original_file_name = matching_files[0]

            # 确定保存的文件名
            if rename_to_short:
                # 短格式: bjfs0540.23.tro.gz
                short_file_name = f"{station_id.lower()}{doy}0.{yy}.tro.gz"
                save_path = save_dir / short_file_name
                print(f"重命名为: {short_file_name}")
            else:
                # 保持原始文件名
                save_path = save_dir / original_file_name

            # 检查文件是否已存在
            if save_path.exists():
                print(f"文件已存在: {save_path}")
                ftp.quit()
                return str(save_path)

            # 下载文件
            with open(save_path, 'wb') as f:
                ftp.retrbinary(f'RETR {original_file_name}', f.write)

            ftp.quit()

            print(f"下载成功: {save_path.name}")
            return str(save_path)

        except Exception as e:
            print(f"下载失败 {station_id}: {e}")
            if 'save_path' in locals() and save_path.exists():
                save_path.unlink()  # 删除不完整的文件
            return None

    def download_euref_ztd(
        self,
        station_id: str,
        date: datetime,
        rename_to_short: bool = True
    ) -> Optional[str]:
        """
        下载欧洲EUREF对流层产品

        URL格式: https://igs.bkg.bund.de/root_ftp/EUREF/products/{GPS_WEEK}/
        文件名: {SITE}0EPNFIN_{YYYYDOY}0000_01D_01H_TRO.TRO.gz
        例如: ASI0EPNFIN_20251170000_01D_01H_TRO.TRO.gz

        Args:
            station_id: 站点ID (4字符，例如: "asi0", "onsa")
            date: 日期
            rename_to_short: 是否重命名为短格式 (例如: asi01170.25.tro.gz)

        Returns:
            下载文件路径
        """
        year = date.year
        doy = date.strftime("%j")  # 年积日
        yy = date.strftime("%y")   # 两位年份

        # 计算GPS周
        gps_epoch = datetime(1980, 1, 6)
        days_since_epoch = (date - gps_epoch).days
        gps_week = days_since_epoch // 7

        # EUREF文件名格式
        # 站点ID通常是4字符，但文件名中可能有后缀0
        station_upper = station_id.upper()
        if not station_upper.endswith('0'):
            station_upper = station_upper[:4] + '0'

        # 构建URL
        file_name = f"{station_upper}EPNFIN_{year}{doy}0000_01D_01H_TRO.TRO.gz"
        url = f"{self.DATA_SOURCES['euref']['url']}/{gps_week}/{file_name}"

        # 保存路径
        save_dir = self.save_dir / "EUREF" / str(year) / doy
        save_dir.mkdir(parents=True, exist_ok=True)

        # 确定保存的文件名
        if rename_to_short:
            # 短格式: asi01170.25.tro.gz
            short_file_name = f"{station_id.lower()}{doy}0.{yy}.tro.gz"
            save_path = save_dir / short_file_name
        else:
            # 保持原始文件名
            save_path = save_dir / file_name

        # 检查文件是否已存在
        if save_path.exists():
            print(f"文件已存在: {save_path}")
            return str(save_path)

        try:
            print(f"正在下载EUREF数据: {station_id.upper()}")
            if rename_to_short:
                print(f"重命名为: {short_file_name}")

            # 使用requests下载
            response = requests.get(url, timeout=60)
            response.raise_for_status()

            # 保存文件
            with open(save_path, 'wb') as f:
                f.write(response.content)

            print(f"下载成功: {save_path.name}")
            return str(save_path)

        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                # 尝试其他可能的文件名格式
                print(f"文件未找到，尝试其他格式...")

                # 尝试不带0后缀的站点名
                alt_station = station_id.upper()[:4]
                alt_file_name = f"{alt_station}0EPNFIN_{year}{doy}0000_01D_01H_TRO.TRO.gz"
                alt_url = f"{self.DATA_SOURCES['euref']['url']}/{gps_week}/{alt_file_name}"

                try:
                    response = requests.get(alt_url, timeout=60)
                    response.raise_for_status()

                    with open(save_path, 'wb') as f:
                        f.write(response.content)

                    print(f"下载成功: {save_path.name}")
                    return str(save_path)
                except:
                    print(f"下载失败 {station_id}: 文件不存在")
                    return None
            else:
                print(f"下载失败 {station_id}: {e}")
                return None

        except Exception as e:
            print(f"下载失败 {station_id}: {e}")
            if save_path.exists():
                save_path.unlink()  # 删除不完整的文件
            return None

    # EUREF分析中心列表
    EUREF_ANALYSIS_CENTERS = [
        'ASI0', 'BEV0', 'BKG0', 'COD0', 'GFZ0', 'IGE0', 'LPT0', 'MUT0',
        'NKG0', 'RGA0', 'ROB0', 'SGO0', 'SUT0', 'UPA0', 'WUT0'
    ]

    def download_euref_analysis_center_files(
        self,
        date: datetime,
        analysis_centers: List[str] = None,
        max_workers: int = 4
    ) -> List[str]:
        """
        下载EUREF分析中心的对流层产品文件

        EUREF数据按分析中心组织，每个分析中心文件包含多个站点的数据。
        URL格式: https://igs.bkg.bund.de/root_ftp/EUREF/products/{GPS_WEEK}/
        文件名: {AC}0EPNFIN_{YYYYDOY}0000_01D_01H_TRO.TRO.gz

        Args:
            date: 日期
            analysis_centers: 要下载的分析中心列表，默认下载所有
            max_workers: 并行下载线程数

        Returns:
            成功下载的文件路径列表
        """
        if analysis_centers is None:
            analysis_centers = self.EUREF_ANALYSIS_CENTERS

        year = date.year
        doy = date.strftime("%j")

        # 计算GPS周
        gps_epoch = datetime(1980, 1, 6)
        days_since_epoch = (date - gps_epoch).days
        gps_week = days_since_epoch // 7

        # 保存目录
        save_dir = self.save_dir / "EUREF" / str(year) / doy
        save_dir.mkdir(parents=True, exist_ok=True)

        downloaded_files = []

        def download_ac_file(ac: str) -> Optional[str]:
            """下载单个分析中心文件"""
            file_name = f"{ac}EPNFIN_{year}{doy}0000_01D_01H_TRO.TRO.gz"
            url = f"{self.DATA_SOURCES['euref']['url']}/{gps_week}/{file_name}"
            save_path = save_dir / file_name

            # 检查文件是否已存在
            if save_path.exists():
                return str(save_path)

            try:
                response = requests.get(url, timeout=60)
                response.raise_for_status()

                with open(save_path, 'wb') as f:
                    f.write(response.content)

                return str(save_path)

            except Exception as e:
                # 静默失败，某些分析中心可能没有数据
                return None

        print(f"正在下载EUREF分析中心文件 (日期: {date.strftime('%Y-%m-%d')}, GPS周: {gps_week})...")

        # 并行下载
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(download_ac_file, ac): ac for ac in analysis_centers}
            for future in as_completed(futures):
                result = future.result()
                if result:
                    downloaded_files.append(result)

        print(f"  成功下载 {len(downloaded_files)}/{len(analysis_centers)} 个分析中心文件")
        return downloaded_files

    def parse_euref_combined_files(
        self,
        file_paths: List[str],
        station_filter: List[str] = None
    ) -> pd.DataFrame:
        """
        解析EUREF分析中心的合并TRO文件

        每个文件包含:
        - +SITE/ID 部分: 站点元数据（ID、坐标等）
        - +TROP/SOLUTION 部分: 对流层数据

        Args:
            file_paths: TRO文件路径列表
            station_filter: 可选的站点ID过滤列表

        Returns:
            DataFrame包含所有站点的ZTD数据和坐标
        """
        all_data = []

        for file_path in file_paths:
            try:
                # 解析单个文件
                df = self._parse_single_euref_file(file_path)
                if not df.empty:
                    all_data.append(df)
            except Exception as e:
                print(f"  解析失败 {Path(file_path).name}: {e}")
                continue

        if not all_data:
            return pd.DataFrame()

        # 合并所有数据
        combined_df = pd.concat(all_data, ignore_index=True)

        # 应用站点过滤
        if station_filter:
            station_filter_upper = [s.upper() for s in station_filter]
            combined_df = combined_df[combined_df['station_id'].str.upper().isin(station_filter_upper)]

        # 去重（同一站点可能在多个分析中心出现）
        # 按站点和时间去重，保留第一个（通常质量较好）
        if not combined_df.empty:
            combined_df = combined_df.drop_duplicates(
                subset=['station_id', 'datetime'],
                keep='first'
            )

        return combined_df

    def _parse_single_euref_file(self, file_path: str) -> pd.DataFrame:
        """
        解析单个EUREF TRO文件

        Args:
            file_path: TRO文件路径

        Returns:
            DataFrame包含站点数据
        """
        # 读取站点元数据
        site_info = {}
        trop_data = []

        try:
            if file_path.endswith('.gz'):
                f = gzip.open(file_path, 'rt')
            else:
                f = open(file_path, 'r')

            in_site_section = False
            in_trop_section = False

            for line in f:
                # 解析SITE/ID部分
                if '+SITE/ID' in line:
                    in_site_section = True
                    continue
                elif '-SITE/ID' in line:
                    in_site_section = False
                    continue

                if in_site_section and not line.startswith('*'):
                    # 解析站点信息行
                    # SINEX格式: STATION PT DOMES T DESCRIPTION... _LONGITUDE _LATITUDE_ _HGT_ELI_ [_HGT_MSL_]
                    # 注意: 经度在纬度前面，且尾部列数因分析中心而异(3或4个数值)
                    parts = line.split()
                    if len(parts) >= 6:
                        station_id = parts[0]
                        try:
                            # 从末尾提取数值列（3或4个）
                            nums = []
                            for p in reversed(parts[1:]):
                                try:
                                    nums.insert(0, float(p))
                                except ValueError:
                                    break
                            # nums: [lon, lat, hgt_eli] 或 [lon, lat, hgt_eli, hgt_msl]
                            if len(nums) >= 3:
                                lon = nums[0]
                                lat = nums[1]
                                height = nums[2]
                                # 经度转换：SINEX使用0-360度，需要转换为-180到180
                                if lon > 180:
                                    lon = lon - 360
                                site_info[station_id] = {
                                    'lat': lat,
                                    'lon': lon,
                                    'height': height
                                }
                        except (ValueError, IndexError):
                            continue

                # 解析TROP/SOLUTION部分
                if '+TROP/SOLUTION' in line:
                    in_trop_section = True
                    continue
                elif '-TROP/SOLUTION' in line:
                    in_trop_section = False
                    continue

                if in_trop_section and not line.startswith('*'):
                    # 解析对流层数据行
                    # 格式: STATION   ___EPOCH______ TROTOT STDDEV ...
                    parts = line.split()
                    if len(parts) >= 4:
                        station_id = parts[0]
                        epoch_str = parts[1]  # 格式: YYYY:DOY:SSSSS
                        try:
                            ztd = float(parts[2])  # TROTOT in mm
                            std = float(parts[3])  # STDDEV in mm

                            # 解析时间
                            epoch_parts = epoch_str.split(':')
                            year = int(epoch_parts[0])
                            doy = int(epoch_parts[1])
                            seconds = int(epoch_parts[2])

                            # 转换为datetime
                            dt = datetime(year, 1, 1) + timedelta(days=doy-1, seconds=seconds)

                            trop_data.append({
                                'station_id': station_id,
                                'datetime': dt,
                                'ztd_mm': ztd,
                                'ztd_std_mm': std
                            })
                        except (ValueError, IndexError):
                            continue

            f.close()

        except Exception as e:
            print(f"  读取文件失败 {file_path}: {e}")
            return pd.DataFrame()

        if not trop_data:
            return pd.DataFrame()

        # 创建DataFrame
        df = pd.DataFrame(trop_data)

        # 添加站点坐标
        df['lat'] = df['station_id'].map(lambda x: site_info.get(x, {}).get('lat', np.nan))
        df['lon'] = df['station_id'].map(lambda x: site_info.get(x, {}).get('lon', np.nan))
        df['height'] = df['station_id'].map(lambda x: site_info.get(x, {}).get('height', np.nan))

        # 移除没有坐标的站点
        df = df.dropna(subset=['lat', 'lon'])

        return df

    def download_and_parse_euref(
        self,
        date: datetime,
        lat_range: Tuple[float, float] = None,
        lon_range: Tuple[float, float] = None
    ) -> pd.DataFrame:
        """
        下载并解析EUREF数据的便捷方法

        Args:
            date: 日期
            lat_range: 纬度范围 (min, max)，可选
            lon_range: 经度范围 (min, max)，可选

        Returns:
            DataFrame包含EUREF ZTD数据
        """
        # 下载分析中心文件
        files = self.download_euref_analysis_center_files(date)

        if not files:
            print("  警告: 未能下载任何EUREF分析中心文件")
            return pd.DataFrame()

        # 解析文件
        df = self.parse_euref_combined_files(files)

        if df.empty:
            print("  警告: EUREF文件解析后无数据")
            return pd.DataFrame()

        # 空间过滤
        if lat_range is not None:
            df = df[(df['lat'] >= lat_range[0]) & (df['lat'] <= lat_range[1])]
        if lon_range is not None:
            df = df[(df['lon'] >= lon_range[0]) & (df['lon'] <= lon_range[1])]

        # 添加数据源标记
        df['source'] = 'euref'

        print(f"  EUREF: {len(df)} 条记录, {df['station_id'].nunique()} 个站点")

        return df

    def parse_igs_whu_file(self, file_path: str) -> pd.DataFrame:
        """
        解析IGS武汉文件（自动识别TRO或ZPD格式）

        TRO格式（新）:
        +TROP/SOLUTION
        *SITE ____EPOCH___ TROTOT STDDEV
         WUHN 23:054:00000 2488.7    4.3
        -TROP/SOLUTION

        ZPD格式（旧）:
        EPOCH YY MM DD HH MM SS    ZTD   SIGMA
        2023 11 01 00 00 00     2345.6   1.2

        Args:
            file_path: .TRO/.zpd文件路径（可能是.gz压缩）

        Returns:
            DataFrame包含时间和ZTD
        """
        data = []

        try:
            # 判断是否是gz文件
            if file_path.endswith('.gz'):
                import gzip
                f = gzip.open(file_path, 'rt')
            else:
                f = open(file_path, 'r')

            # 判断文件格式（TRO或ZPD）
            is_tro_format = '.TRO' in file_path.upper() or '.tro' in file_path

            if is_tro_format:
                # 解析TRO格式
                in_solution_section = False

                for line in f:
                    # 检查是否进入TROP/SOLUTION部分
                    if '+TROP/SOLUTION' in line:
                        in_solution_section = True
                        continue
                    elif '-TROP/SOLUTION' in line:
                        in_solution_section = False
                        break

                    # 解析数据行
                    if in_solution_section and not line.startswith('*'):
                        parts = line.split()
                        if len(parts) >= 4:
                            try:
                                site = parts[0]
                                epoch = parts[1]  # YY:DOY:SSSSS
                                ztd_mm = float(parts[2])  # mm
                                sigma_mm = float(parts[3])  # mm

                                # 解析epoch
                                epoch_parts = epoch.split(':')
                                if len(epoch_parts) == 3:
                                    year_val = int(epoch_parts[0])
                                    # 判断是四位年份还是两位年份
                                    if year_val >= 1900:
                                        # 四位年份 (例如: 2025)
                                        year = year_val
                                    elif year_val >= 80:
                                        # 两位年份 1980-1999
                                        year = 1900 + year_val
                                    else:
                                        # 两位年份 2000-2079
                                        year = 2000 + year_val
                                    doy = int(epoch_parts[1])  # 年积日
                                    seconds = int(epoch_parts[2])  # 秒

                                    # 转换为datetime
                                    base_date = datetime(year, 1, 1)
                                    dt = base_date + timedelta(days=doy-1, seconds=seconds)

                                    data.append({
                                        'datetime': dt,
                                        'ztd_mm': ztd_mm,
                                        'sigma_mm': sigma_mm,
                                        'station_id': site
                                    })

                            except (ValueError, IndexError) as e:
                                continue

            else:
                # 解析ZPD格式（旧格式）
                for line in f:
                    if line.startswith('*') or line.startswith('#') or not line.strip():
                        continue

                    parts = line.split()
                    if len(parts) >= 9:
                        try:
                            year = int(parts[0])
                            month = int(parts[1])
                            day = int(parts[2])
                            hour = int(parts[3])
                            minute = int(parts[4])
                            second = int(parts[5])
                            ztd_m = float(parts[6])  # 米
                            sigma_m = float(parts[7])  # 米

                            dt = datetime(year, month, day, hour, minute, second)

                            data.append({
                                'datetime': dt,
                                'ztd_mm': ztd_m * 1000,  # 转换为mm
                                'sigma_mm': sigma_m * 1000
                            })
                        except (ValueError, IndexError):
                            continue

            f.close()

        except Exception as e:
            print(f"解析IGS武汉文件失败 {file_path}: {e}")

        return pd.DataFrame(data)

    def download_station_ztd(
        self,
        station_id: str,
        year: int
    ) -> str:
        """
        [已弃用] 下载单个站点的ZTD数据 (原NGL方法)

        注意: 此方法已弃用，NGL数据源不再支持。
        请使用 download_igs_whu_ztd, download_euref_ztd 等特定数据源方法。

        Args:
            station_id: 站点ID (例如: "P101")
            year: 年份

        Returns:
            下载文件路径
        """
        print("警告: download_station_ztd 方法已弃用，NGL数据源不再支持")
        print("请使用特定数据源方法: download_igs_whu_ztd, download_euref_ztd 等")
        return None

    def parse_ztd_file(self, trop_file: str) -> pd.DataFrame:
        """
        [已弃用] 解析NGL ZTD文件

        注意: 此方法已弃用，NGL数据源不再支持。
        请使用 parse_igs_whu_file, parse_euref_file 等特定数据源解析方法。

        Args:
            trop_file: .trop文件路径

        Returns:
            空DataFrame
        """
        print("警告: parse_ztd_file 方法已弃用，NGL数据源不再支持")
        print("请使用特定数据源解析方法: parse_igs_whu_file 等")
        return pd.DataFrame()

    def ztd_to_pwv(
        self,
        ztd_mm: float,
        latitude: float = 45.0,
        height_m: float = 0.0,
        temperature_k: Optional[float] = None,
        pressure_pa: Optional[float] = None
    ) -> float:
        """
        将ZTD(天顶总延迟)转换为PWV(可降水量)

        物理公式:
        1. ZHD = 0.0022768 * P / f(φ,H)  (Saastamoinen静力延迟)
        2. ZWD = ZTD - ZHD  (湿延迟)
        3. PWV = Π * ZWD, 其中 Π = 10^6 / (ρ_w * R_v * ((k3/Tm) + k2'))

        参考: Bevis et al. (1992, 1994)

        Args:
            ztd_mm: 天顶总延迟(mm)
            latitude: 纬度(度)
            height_m: 站点高度(米)
            temperature_k: 地面温度(K), 如果为None则使用默认值
            pressure_pa: 地面气压(Pa), 如果为None则使用默认值

        Returns:
            PWV (kg/m²)
        """
        # 默认值
        if temperature_k is None:
            temperature_k = 288.0
        if pressure_pa is None:
            pressure_pa = 101325.0

        # 转换为hPa
        pressure_hpa = pressure_pa / 100.0

        # 1. 计算静力延迟 ZHD (Saastamoinen模型)
        # f(φ,H) = 1 - 0.00266*cos(2φ) - 0.00028*H
        phi_rad = np.radians(latitude)
        f_factor = 1.0 - 0.00266 * np.cos(2 * phi_rad) - 0.00028 * (height_m / 1000.0)
        zhd_mm = 2.2768 * pressure_hpa / f_factor

        # 2. 计算湿延迟 ZWD
        zwd_mm = ztd_mm - zhd_mm

        # 如果ZWD为负值，说明可能有错误，返回0
        if zwd_mm < 0:
            print(f"警告: ZWD为负值 ({zwd_mm:.2f} mm), 返回PWV=0")
            return 0.0

        # 3. 计算加权平均温度 Tm (Bevis公式)
        # Tm = 70.2 + 0.72*Ts (K)
        Tm = 70.2 + 0.72 * temperature_k  # K

        # 4. 物理常数
        rho_w = 1000.0  # 水密度 kg/m³
        Rv = 461.5      # 水汽气体常数 J/(kg·K)
        # Bevis公式中的常数（转换为Pa单位）
        k2_prime = 0.221  # K/Pa (= 22.1 K/hPa)
        k3 = 3.739e3    # K²/Pa (= 3.739×10^5 K²/hPa)

        # 5. 计算转换系数 Π
        Pi = 1e6 / (rho_w * Rv * (k3 / Tm + k2_prime))

        # 6. 计算PWV
        # Π的单位是无量纲的，ZWD单位是mm，所以PWV单位也是mm
        pwv = Pi * zwd_mm  # mm

        return pwv

    def download_australia_ztd(
        self,
        date: datetime,
        region: str = "gar1",
        hours: Optional[List[int]] = None,
        max_workers: int = 8,
        show_progress: bool = True
    ) -> List[str]:
        """
        下载澳大利亚对流层延迟数据（优化版：支持并行下载和进度条）

        文件格式: cost_h_t_YYYYMMDDHHMM_YYYYMMDDHHMM_mult_{region}.dat
        例如: cost_h_t_202511291100_202511291159_mult_gar1.dat

        数据来源: https://ga-gnss-products-v1.s3.amazonaws.com/public/{gps_week}/

        Args:
            date: 日期
            region: 区域代码 ('gar1': APREF+IGS, 'gau1': ARGN, 'gau2': NSW, 'gau3': VIC)
            hours: 要下载的小时列表(0-23)，如果为None则下载全天24小时（默认）
            max_workers: 最大并发下载线程数（默认8）
            show_progress: 是否显示进度条

        Returns:
            下载成功的文件路径列表
        """
        year = date.year
        month = date.month
        day = date.day

        # 如果未指定小时，则下载全天
        if hours is None:
            hours = list(range(24))

        # 计算GPS周（只需计算一次）
        # 注意：澳大利亚服务器的GPS周比标准GPS周晚1周
        gps_epoch = datetime(1980, 1, 6)
        days_since_epoch = (date - gps_epoch).days
        gps_week = days_since_epoch // 7  # 标准GPS周计算

        print(f"澳大利亚数据下载")
        print(f"日期: {date.strftime('%Y-%m-%d')}, GPS周: {gps_week}, 区域: {region}")
        print(f"下载小时数: {len(hours)}, 并发线程: {max_workers}")
        print("-" * 60)

        def download_single_hour(hour: int) -> Optional[str]:
            """下载单个小时的数据"""
            dt_start = datetime(year, month, day, hour, 0)
            dt_end = datetime(year, month, day, hour, 59)

            start_str = dt_start.strftime("%Y%m%d%H%M")
            end_str = dt_end.strftime("%Y%m%d%H%M")

            file_name = f"cost_h_t_{start_str}_{end_str}_mult_{region}.dat"

            # 正确的URL结构: /public/{gps_week}/{file_name}
            url = f"{self.DATA_SOURCES['australia']['url']}/public/{gps_week}/{file_name}"

            # 保存路径
            save_path = self.save_dir / "Australia" / str(year) / f"{month:02d}" / f"{day:02d}" / file_name
            save_path.parent.mkdir(parents=True, exist_ok=True)

            # 检查文件是否已存在
            if save_path.exists():
                return str(save_path)

            # 下载文件（带重试）
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    response = requests.get(url, timeout=60)
                    response.raise_for_status()

                    # 保存文件
                    with open(save_path, 'wb') as f:
                        f.write(response.content)

                    return str(save_path)

                except requests.exceptions.RequestException as e:
                    if attempt < max_retries - 1:
                        continue
                    else:
                        if save_path.exists():
                            save_path.unlink()  # 删除不完整的文件
                        return None

        # 并行下载（带进度条）
        downloaded_files = []

        if TQDM_AVAILABLE and show_progress:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(download_single_hour, hour): hour for hour in hours}

                with tqdm(total=len(hours), desc="下载澳大利亚数据", unit="文件") as pbar:
                    for future in as_completed(futures):
                        file_path = future.result()
                        if file_path:
                            downloaded_files.append(file_path)
                        pbar.update(1)
        else:
            # 无进度条版本
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_hour = {
                    executor.submit(download_single_hour, hour): hour
                    for hour in hours
                }

                for future in as_completed(future_to_hour):
                    hour = future_to_hour[future]
                    file_path = future.result()
                    if file_path:
                        downloaded_files.append(file_path)
                        print(f"[成功] {hour:02d}:00")
                    else:
                        print(f"[失败] {hour:02d}:00")

        print("-" * 60)
        print(f"下载完成: {len(downloaded_files)}/{len(hours)} 个文件 ({len(downloaded_files)/len(hours)*100:.1f}%)")

        return downloaded_files

    def parse_australia_file(self, dat_file: str) -> pd.DataFrame:
        """
        解析澳大利亚对流层延迟数据文件（COST-716格式）

        COST-716格式是分块的，每个站点一个数据块：
        - 头部：COST-716 V2.2
        - 站点信息：SITE_ID
        - 位置：LAT LON HEIGHT
        - 日期范围：DATE_START DATE_END
        - 数据行：HH MM SS FLAGS ZTD(mm) SIGMA(mm) ...

        Args:
            dat_file: .dat文件路径

        Returns:
            DataFrame包含站点、位置和ZTD
        """
        data = []

        try:
            with open(dat_file, 'r') as f:
                lines = f.readlines()

            i = 0
            while i < len(lines):
                line = lines[i].strip()

                # 查找COST-716块的开始
                if line.startswith('COST-716'):
                    # 解析站点信息（下一行）
                    if i + 1 < len(lines):
                        station_line = lines[i + 1].split()
                        if len(station_line) >= 2:
                            station_id = station_line[0]

                            # 跳过接收机信息行
                            i += 3

                            # 解析位置信息（LAT LON HEIGHT）
                            if i < len(lines):
                                pos_parts = lines[i].split()
                                if len(pos_parts) >= 3:
                                    try:
                                        lat = float(pos_parts[0])
                                        lon = float(pos_parts[1])
                                        height = float(pos_parts[2])

                                        # 如果位置是-999（无效值），设为NaN但继续解析
                                        if lat < -900 or lon < -900:
                                            lat = float('nan')
                                            lon = float('nan')
                                        if height < -900:
                                            height = float('nan')

                                        # 解析日期范围（下一行）
                                        i += 1
                                        if i < len(lines):
                                            date_line = lines[i].strip()
                                            # 格式：29-Nov-2025 11:00:00     29-Nov-2025 12:30:00
                                            date_parts = date_line.split()
                                            if len(date_parts) >= 2:
                                                # 解析起始日期
                                                date_str = date_parts[0]
                                                time_str = date_parts[1]
                                                try:
                                                    base_date = datetime.strptime(
                                                        f"{date_str} {time_str}",
                                                        "%d-%b-%Y %H:%M:%S"
                                                    )

                                                    # 跳过到数据行
                                                    # 从date_line之后：+1(processing info), +2(sample), +3(flags), +4(n_epochs), +5(data)
                                                    i += 5

                                                    # 读取数据行直到下一个COST-716块或文件结束
                                                    while i < len(lines):
                                                        data_line = lines[i].strip()

                                                        # 检查是否是新的COST-716块
                                                        if data_line.startswith('COST-716') or data_line.startswith('---'):
                                                            break

                                                        # 解析数据行（格式：HH MM SS FLAGS ZTD SIGMA ...）
                                                        parts = data_line.split()
                                                        if len(parts) >= 6 and parts[0].isdigit():
                                                            try:
                                                                hour = int(parts[0])
                                                                minute = int(parts[1])
                                                                second = int(parts[2])
                                                                ztd_mm = float(parts[4])  # mm
                                                                sigma_mm = float(parts[5])  # mm

                                                                # 构建完整时间
                                                                dt = datetime(
                                                                    base_date.year,
                                                                    base_date.month,
                                                                    base_date.day,
                                                                    hour, minute, second
                                                                )

                                                                # 数据验证
                                                                if 1000 <= ztd_mm <= 3000 and 0 < sigma_mm < 100:
                                                                    data.append({
                                                                        'station_id': station_id,
                                                                        'lat': lat,
                                                                        'lon': lon,
                                                                        'height': height,
                                                                        'datetime': dt,
                                                                        'ztd_mm': ztd_mm,
                                                                        'sigma_mm': sigma_mm
                                                                    })

                                                            except (ValueError, IndexError):
                                                                pass

                                                        i += 1
                                                    continue

                                                except ValueError:
                                                    pass

                                    except (ValueError, IndexError):
                                        pass

                i += 1

        except Exception as e:
            print(f"解析澳大利亚文件失败 {dat_file}: {e}")

        return pd.DataFrame(data)

    def parse_australia_files_batch(
        self,
        dat_files: List[str],
        max_workers: int = 5
    ) -> pd.DataFrame:
        """
        批量解析澳大利亚数据文件（并行处理）

        Args:
            dat_files: .dat文件路径列表
            max_workers: 最大并发线程数

        Returns:
            合并后的DataFrame
        """
        print(f"批量解析 {len(dat_files)} 个澳大利亚数据文件...")

        all_data = []

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_file = {
                executor.submit(self.parse_australia_file, f): f
                for f in dat_files
            }

            for future in as_completed(future_to_file):
                file_path = future_to_file[future]
                try:
                    df = future.result()
                    if not df.empty:
                        all_data.append(df)
                        file_name = Path(file_path).name
                        print(f"  [解析] {file_name}: {len(df)} 条数据")
                except Exception as e:
                    print(f"  [失败] {Path(file_path).name}: {e}")

        if all_data:
            df_combined = pd.concat(all_data, ignore_index=True)
            print(f"合并完成: 总计 {len(df_combined)} 条数据")

            # 统计信息
            if not df_combined.empty:
                n_stations = df_combined['station_id'].nunique()
                time_range = (df_combined['datetime'].min(), df_combined['datetime'].max())
                print(f"站点数: {n_stations}")
                print(f"时间范围: {time_range[0]} 至 {time_range[1]}")

            return df_combined
        else:
            return pd.DataFrame()

    def download_ucar_pwv(
        self,
        date: datetime,
        hours: Optional[List[int]] = None,
        max_workers: int = 6,
        show_progress: bool = True
    ) -> List[str]:
        """
        下载UCAR COSMIC的PWV数据（优化版：支持进度条）

        数据源: https://data.cosmic.ucar.edu/suominet/postProcess/pwvConus/
        特点:
        - 无需认证，直接下载
        - 提供已处理的PWV数据（无需从ZTD计算）
        - 时间间隔: 30分钟
        - 覆盖区域: 美国CONUS（Continental US）

        URL格式: https://data.cosmic.ucar.edu/suominet/postProcess/pwvConus/yYYYY/SUOd_YYYY.DDD.HH.PWV

        文件格式:
        Site PWVmidTim    Duration PW   FMerr Wdelay Mdelay Tdelay  KFAC  Press  Temp  Rhum Ddelay Mf Kf
        SSSS YYYMMDD/HHMM   MIN   [mm]   [mm]  [mm]  [mm]   [mm]   d.ddd  [mbar]  [c]   [%]   [mm]  C C

        Args:
            date: 下载日期
            hours: 下载的小时列表，默认[0, 6, 12, 18]（UCAR提供的时次）
            max_workers: 最大并发下载数（默认6）
            show_progress: 是否显示进度条

        Returns:
            下载的文件路径列表
        """
        if hours is None:
            hours = [0, 6, 12, 18]  # UCAR提供的标准时次

        year = date.year
        doy = date.strftime("%j")

        print(f"正在下载UCAR COSMIC PWV数据: {date.strftime('%Y-%m-%d')}")
        print(f"下载时次: {hours}, 并发线程: {max_workers}")

        # 保存目录
        save_dir = self.save_dir / "UCAR_COSMIC" / str(year) / doy
        save_dir.mkdir(parents=True, exist_ok=True)

        def download_single_hour(hour: int, max_retries: int = 3) -> Optional[str]:
            """下载单个小时的数据"""
            file_name = f"SUOd_{year}.{doy}.{hour:02d}.PWV"
            url = f"https://data.cosmic.ucar.edu/suominet/postProcess/pwvConus/y{year}/{file_name}"
            save_path = save_dir / file_name

            # 检查是否已存在
            if save_path.exists():
                return str(save_path)

            # 下载文件
            for attempt in range(max_retries):
                try:
                    response = requests.get(url, timeout=30)

                    if response.status_code == 200:
                        with open(save_path, 'wb') as f:
                            f.write(response.content)
                        return str(save_path)
                    elif response.status_code == 404:
                        return None
                    else:
                        if attempt < max_retries - 1:
                            continue
                        return None

                except requests.exceptions.RequestException as e:
                    if attempt < max_retries - 1:
                        continue
                    else:
                        return None

            return None

        # 并行下载（带进度条）
        downloaded_files = []

        if TQDM_AVAILABLE and show_progress:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(download_single_hour, hour): hour for hour in hours}

                with tqdm(total=len(hours), desc="下载UCAR PWV数据", unit="文件") as pbar:
                    for future in as_completed(futures):
                        file_path = future.result()
                        if file_path:
                            downloaded_files.append(file_path)
                        pbar.update(1)
        else:
            # 无进度条版本
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_hour = {
                    executor.submit(download_single_hour, hour): hour
                    for hour in hours
                }

                for future in as_completed(future_to_hour):
                    hour = future_to_hour[future]
                    file_path = future.result()
                    if file_path:
                        downloaded_files.append(file_path)
                        print(f"[成功] {hour:02d}:00")
                    else:
                        print(f"[跳过] {hour:02d}:00 - 文件不存在")

        print("-" * 60)
        print(f"下载完成: {len(downloaded_files)}/{len(hours)} 个文件 ({len(downloaded_files)/len(hours)*100:.1f}%)")

        return downloaded_files

    def parse_ucar_pwv_file(self, pwv_file: str) -> pd.DataFrame:
        """
        解析UCAR COSMIC PWV文件

        文件格式:
        Site PWVmidTim    Duration PW   FMerr Wdelay Mdelay Tdelay  KFAC  Press  Temp  Rhum Ddelay Mf Kf
        1LSU 20150101/0015   30   12.5    0.4   80.4  -99.9 2419.7  6.428 1026.0   7.0  70.5 2339.3 -99.9 -99.9 -99.9 E B

        字段说明:
        - Site: 站点ID (4字符)
        - PWVmidTim: 时间 YYYYMMDD/HHMM
        - Duration: 时长（分钟）
        - PW: PWV值 (mm)
        - FMerr: PWV误差 (mm)
        - Wdelay: 湿延迟 (mm)
        - Tdelay: 总延迟 (mm，即ZTD)
        - Press: 气压 (mbar)
        - Temp: 温度 (°C)
        - Rhum: 相对湿度 (%)

        Args:
            pwv_file: .PWV文件路径

        Returns:
            DataFrame包含站点、时间、PWV和ZTD
        """
        data = []

        try:
            with open(pwv_file, 'r') as f:
                for line in f:
                    # 跳过头部和空行
                    if line.startswith('Site') or line.startswith('SSSS') or not line.strip():
                        continue

                    parts = line.split()
                    if len(parts) >= 8:
                        try:
                            station_id = parts[0]
                            time_str = parts[1]  # YYYYMMDD/HHMM
                            duration_min = int(parts[2])
                            pwv_mm = float(parts[3])
                            pwv_err_mm = float(parts[4])
                            wet_delay_mm = float(parts[5])

                            # 总延迟（ZTD）在第7列
                            ztd_mm = float(parts[7]) if len(parts) >= 8 else -99.9

                            # 气象数据（可选）
                            pressure_mbar = float(parts[9]) if len(parts) >= 10 else -99.9
                            temp_c = float(parts[10]) if len(parts) >= 11 else -99.9
                            humidity_pct = float(parts[11]) if len(parts) >= 12 else -99.9

                            # 解析时间 YYYYMMDD/HHMM
                            date_part, time_part = time_str.split('/')
                            year = int(date_part[:4])
                            month = int(date_part[4:6])
                            day = int(date_part[6:8])
                            hour = int(time_part[:2])
                            minute = int(time_part[2:4])

                            dt = datetime(year, month, day, hour, minute)

                            # 跳过无效值
                            if pwv_mm < -90 or ztd_mm < -90:
                                continue

                            data.append({
                                'station_id': station_id,
                                'datetime': dt,
                                'pwv_mm': pwv_mm,
                                'pwv_err_mm': pwv_err_mm,
                                'wet_delay_mm': wet_delay_mm,
                                'ztd_mm': ztd_mm,
                                'duration_min': duration_min,
                                'pressure_mbar': pressure_mbar if pressure_mbar > -90 else None,
                                'temp_c': temp_c if temp_c > -90 else None,
                                'humidity_pct': humidity_pct if humidity_pct > -90 else None
                            })

                        except (ValueError, IndexError):
                            continue

        except Exception as e:
            print(f"解析UCAR PWV文件失败 {pwv_file}: {e}")

        return pd.DataFrame(data)

    def parse_ucar_pwv_files_batch(
        self,
        pwv_files: List[str],
        max_workers: int = 4
    ) -> pd.DataFrame:
        """
        批量解析UCAR COSMIC PWV文件（并行处理）

        Args:
            pwv_files: .PWV文件路径列表
            max_workers: 最大并发线程数

        Returns:
            合并后的DataFrame
        """
        print(f"批量解析 {len(pwv_files)} 个UCAR COSMIC PWV文件...")

        all_data = []

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_file = {
                executor.submit(self.parse_ucar_pwv_file, f): f
                for f in pwv_files
            }

            for future in as_completed(future_to_file):
                file_path = future_to_file[future]
                try:
                    df = future.result()
                    if not df.empty:
                        all_data.append(df)
                        file_name = Path(file_path).name
                        print(f"  [解析] {file_name}: {len(df)} 条数据")
                except Exception as e:
                    print(f"  [失败] {Path(file_path).name}: {e}")

        if all_data:
            df_combined = pd.concat(all_data, ignore_index=True)
            print(f"合并完成: 总计 {len(df_combined)} 条数据")

            # 统计信息
            if not df_combined.empty:
                n_stations = df_combined['station_id'].nunique()
                time_range = (df_combined['datetime'].min(), df_combined['datetime'].max())
                print(f"站点数: {n_stations}")
                print(f"时间范围: {time_range[0]} 至 {time_range[1]}")

                if 'pwv_mm' in df_combined.columns:
                    pwv_range = (df_combined['pwv_mm'].min(), df_combined['pwv_mm'].max())
                    print(f"PWV范围: {pwv_range[0]:.2f} - {pwv_range[1]:.2f} mm")

            return df_combined
        else:
            return pd.DataFrame()

    def process_station_to_pwv(
        self,
        station_id: str,
        year: int,
        station_lat: Optional[float] = None,
        station_lon: Optional[float] = None,
        station_height: Optional[float] = None,
        use_era5_meteo: bool = True
    ) -> pd.DataFrame:
        """
        处理单个站点: 下载ZTD并转换为PWV (使用ERA5气象数据)

        Args:
            station_id: 站点ID
            year: 年份
            station_lat: 站点纬度 (如果use_era5_meteo=True则必需)
            station_lon: 站点经度 (如果use_era5_meteo=True则必需)
            station_height: 站点高度(米)
            use_era5_meteo: 是否使用ERA5气象数据计算PWV

        Returns:
            包含时间和PWV的DataFrame
        """
        # 1. 下载ZTD数据
        trop_file = self.download_station_ztd(station_id, year)
        if trop_file is None:
            return pd.DataFrame()

        # 2. 解析文件
        df = self.parse_ztd_file(trop_file)

        if df.empty:
            return df

        # 3. 转换为PWV
        if use_era5_meteo and station_lat is not None and station_lon is not None:
            print(f"使用ERA5气象数据计算PWV for {station_id}")

            # 为每个时间点获取ERA5气象数据并计算PWV
            pwv_values = []
            for idx, row in df.iterrows():
                try:
                    # 获取ERA5气象数据
                    meteo = self.era5_provider.get_meteo_data(
                        lat=station_lat,
                        lon=station_lon,
                        time=row['datetime']
                    )

                    # 计算PWV
                    pwv = self.ztd_to_pwv(
                        ztd_mm=row['ztd_mm'],
                        latitude=station_lat,
                        height_m=station_height if station_height else 0.0,
                        temperature_k=meteo['temperature_k'],
                        pressure_pa=meteo['pressure_pa']
                    )
                    pwv_values.append(pwv)

                except Exception as e:
                    print(f"警告: PWV计算失败 at {row['datetime']}: {e}")
                    # 使用默认气象值
                    pwv = self.ztd_to_pwv(
                        ztd_mm=row['ztd_mm'],
                        latitude=station_lat,
                        height_m=station_height if station_height else 0.0
                    )
                    pwv_values.append(pwv)

            df['pwv'] = pwv_values
        else:
            # 使用默认气象值
            print(f"使用默认气象值计算PWV for {station_id}")
            df['pwv'] = df['ztd_mm'].apply(
                lambda ztd: self.ztd_to_pwv(
                    ztd,
                    latitude=station_lat if station_lat else 45.0,
                    height_m=station_height if station_height else 0.0
                )
            )

        # 4. 添加站点信息
        df['station_id'] = station_id
        if station_lat is not None:
            df['lat'] = station_lat
        if station_lon is not None:
            df['lon'] = station_lon
        if station_height is not None:
            df['height'] = station_height

        return df

    def batch_download_and_process(
        self,
        station_list: List[str],
        year: int,
        max_workers: int = 10
    ) -> pd.DataFrame:
        """
        批量下载和处理多个站点

        Args:
            station_list: 站点ID列表
            year: 年份
            max_workers: 并发线程数

        Returns:
            所有站点的PWV DataFrame
        """
        all_data = []

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有任务
            future_to_station = {
                executor.submit(self.process_station_to_pwv, sid, year): sid
                for sid in station_list
            }

            # 收集结果
            for future in as_completed(future_to_station):
                station_id = future_to_station[future]
                try:
                    df = future.result()
                    if not df.empty:
                        all_data.append(df)
                        print(f"[PASS] 完成: {station_id}, {len(df)} 条数据")
                except Exception as e:
                    print(f"[FAIL] 失败: {station_id}, {e}")

        # 合并所有数据
        if all_data:
            df_all = pd.concat(all_data, ignore_index=True)
            print(f"\n总计: {len(df_all)} 条PWV数据")
            return df_all
        else:
            return pd.DataFrame()

    def quality_check_ztd(
        self,
        df: pd.DataFrame,
        ztd_min: float = 1500.0,
        ztd_max: float = 3000.0,
        sigma_max: float = 50.0,
        remove_outliers: bool = True
    ) -> pd.DataFrame:
        """
        ZTD数据质量检查

        Args:
            df: 包含ztd_mm和sigma_mm的DataFrame
            ztd_min: ZTD最小值(mm)
            ztd_max: ZTD最大值(mm)
            sigma_max: 最大标准差(mm)
            remove_outliers: 是否移除异常值

        Returns:
            质量检查后的DataFrame
        """
        if df.empty:
            return df

        n_original = len(df)

        # 检查必需列
        if 'ztd_mm' not in df.columns:
            print("警告: DataFrame缺少ztd_mm列")
            return df

        # 1. 范围检查
        valid_range = (df['ztd_mm'] >= ztd_min) & (df['ztd_mm'] <= ztd_max)
        n_range_fail = (~valid_range).sum()

        # 2. 标准差检查
        if 'sigma_mm' in df.columns:
            valid_sigma = (df['sigma_mm'] <= sigma_max) | df['sigma_mm'].isna()
            n_sigma_fail = (~valid_sigma).sum()
            valid_mask = valid_range & valid_sigma
        else:
            valid_mask = valid_range
            n_sigma_fail = 0

        # 3. 统计异常值
        if remove_outliers and 'ztd_mm' in df.columns:
            # 使用3-sigma规则检测异常值
            mean_ztd = df.loc[valid_mask, 'ztd_mm'].mean()
            std_ztd = df.loc[valid_mask, 'ztd_mm'].std()
            outlier_mask = np.abs(df['ztd_mm'] - mean_ztd) > 3 * std_ztd
            n_outliers = outlier_mask.sum()
            valid_mask = valid_mask & ~outlier_mask
        else:
            n_outliers = 0

        # 输出质量报告
        print(f"\n质量检查报告:")
        print(f"  原始数据: {n_original} 条")
        print(f"  范围异常: {n_range_fail} 条 (ZTD不在{ztd_min}-{ztd_max}mm)")
        if 'sigma_mm' in df.columns:
            print(f"  精度异常: {n_sigma_fail} 条 (sigma>{sigma_max}mm)")
        if remove_outliers:
            print(f"  统计异常: {n_outliers} 条 (3-sigma规则)")
        print(f"  有效数据: {valid_mask.sum()} 条 ({valid_mask.sum()/n_original*100:.1f}%)")

        if remove_outliers:
            return df[valid_mask].copy()
        else:
            # 添加质量标记列
            df['quality_flag'] = valid_mask
            return df

    def spatial_match_with_fy3g(
        self,
        gnss_df: pd.DataFrame,
        fy3g_lat: np.ndarray,
        fy3g_lon: np.ndarray,
        fy3g_time: datetime,
        max_distance_km: float = 30.0,
        max_time_hours: float = 3.0
    ) -> Tuple[np.ndarray, pd.DataFrame]:
        """
        GNSS数据与FY3G网格的空间匹配

        Args:
            gnss_df: GNSS数据DataFrame (需包含lat, lon, datetime, ztd_mm或pwv_mm)
            fy3g_lat: FY3G纬度网格 (1D array)
            fy3g_lon: FY3G经度网格 (1D array)
            fy3g_time: FY3G观测时间
            max_distance_km: 最大匹配距离(km)
            max_time_hours: 最大时间差(小时)

        Returns:
            (gnss_mask, matched_gnss):
                - gnss_mask: 布尔数组(nlat, nlon)，标记有GNSS观测的像素
                - matched_gnss: 匹配的GNSS数据DataFrame
        """
        if gnss_df.empty:
            print("警告: GNSS数据为空")
            gnss_mask = np.zeros((len(fy3g_lat), len(fy3g_lon)), dtype=bool)
            return gnss_mask, pd.DataFrame()

        # 1. 时间筛选
        time_diff = np.abs((gnss_df['datetime'] - fy3g_time).dt.total_seconds() / 3600)
        time_valid = time_diff <= max_time_hours
        gnss_filtered = gnss_df[time_valid].copy()

        if gnss_filtered.empty:
            print(f"警告: 时间窗口±{max_time_hours}h内无GNSS数据")
            gnss_mask = np.zeros((len(fy3g_lat), len(fy3g_lon)), dtype=bool)
            return gnss_mask, pd.DataFrame()

        print(f"时间筛选: {len(gnss_filtered)}/{len(gnss_df)} 条数据在±{max_time_hours}h内")

        # 2. 空间匹配
        # 创建FY3G网格点
        fy3g_points = np.array([
            [lat, lon]
            for lat in fy3g_lat
            for lon in fy3g_lon
        ])

        # 创建GNSS站点KD树
        gnss_points = gnss_filtered[['lat', 'lon']].values
        tree = cKDTree(gnss_points)

        # 查询最近的GNSS站点
        # 使用球面距离近似: 1度 ≈ 111km
        max_distance_deg = max_distance_km / 111.0
        distances, indices = tree.query(fy3g_points, distance_upper_bound=max_distance_deg)

        # 3. 创建匹配掩码
        gnss_mask_flat = distances < max_distance_deg
        gnss_mask = gnss_mask_flat.reshape(len(fy3g_lat), len(fy3g_lon))

        # 4. 提取匹配的GNSS数据
        matched_indices = indices[gnss_mask_flat]
        matched_gnss = gnss_filtered.iloc[matched_indices].copy()
        matched_gnss['match_distance_km'] = distances[gnss_mask_flat] * 111.0

        print(f"空间匹配: {gnss_mask.sum()} 个FY3G像素匹配到GNSS站点")
        print(f"  匹配距离: {matched_gnss['match_distance_km'].min():.1f} - "
              f"{matched_gnss['match_distance_km'].max():.1f} km")

        return gnss_mask, matched_gnss

    def download_gnss_for_fy3g_overpass(
        self,
        fy3g_file: str,
        data_sources: List[str] = ['igs_whu', 'australia', 'euref'],
        time_window_hours: float = 3.0,
        spatial_buffer_deg: float = 5.0
    ) -> pd.DataFrame:
        """
        根据FY3G过境自动下载GNSS数据

        Args:
            fy3g_file: FY3G数据文件路径
            data_sources: 数据源列表
            time_window_hours: 时间窗口(小时)
            spatial_buffer_deg: 空间缓冲区(度)

        Returns:
            合并的GNSS数据DataFrame
        """
        print(f"\n{'='*60}")
        print(f"FY3G过境自动下载GNSS数据")
        print(f"{'='*60}")

        # 1. 读取FY3G数据获取时空信息
        try:
            ds = xr.open_dataset(fy3g_file)
            fy3g_time = pd.to_datetime(ds.time.values)
            fy3g_lat = ds.lat.values
            fy3g_lon = ds.lon.values

            lat_min, lat_max = fy3g_lat.min(), fy3g_lat.max()
            lon_min, lon_max = fy3g_lon.min(), fy3g_lon.max()

            ds.close()

            print(f"\nFY3G过境信息:")
            print(f"  时间: {fy3g_time}")
            print(f"  纬度范围: {lat_min:.2f}° - {lat_max:.2f}°")
            print(f"  经度范围: {lon_min:.2f}° - {lon_max:.2f}°")

        except Exception as e:
            print(f"错误: 无法读取FY3G文件 - {e}")
            return pd.DataFrame()

        # 2. 扩展空间范围
        lat_min -= spatial_buffer_deg
        lat_max += spatial_buffer_deg
        lon_min -= spatial_buffer_deg
        lon_max += spatial_buffer_deg

        print(f"\n扩展后范围(±{spatial_buffer_deg}°):")
        print(f"  纬度: {lat_min:.2f}° - {lat_max:.2f}°")
        print(f"  经度: {lon_min:.2f}° - {lon_max:.2f}°")

        # 3. 根据数据源下载
        all_gnss_data = []

        for source in data_sources:
            print(f"\n{'='*60}")
            print(f"数据源: {source.upper()}")
            print(f"{'='*60}")

            try:
                if source == 'australia':
                    # 澳大利亚数据 - 下载全天24小时
                    files = self.download_australia_ztd(
                        date=fy3g_time,
                        region='gar1',
                        hours=None  # 下载全天
                    )
                    if files:
                        df = self.parse_australia_files_batch(files)
                        if not df.empty:
                            # 空间筛选
                            mask = (
                                (df['lat'] >= lat_min) & (df['lat'] <= lat_max) &
                                (df['lon'] >= lon_min) & (df['lon'] <= lon_max)
                            )
                            df_filtered = df[mask]
                            print(f"  空间筛选: {len(df_filtered)}/{len(df)} 条数据在范围内")

                            if not df_filtered.empty:
                                df_filtered['source'] = 'australia'
                                all_gnss_data.append(df_filtered)

                elif source == 'igs_whu':
                    # IGS武汉 - 需要站点列表
                    if self.stations_file.exists():
                        stations_df = pd.read_csv(self.stations_file)
                        # 空间筛选站点
                        mask = (
                            (stations_df['lat'] >= lat_min) & (stations_df['lat'] <= lat_max) &
                            (stations_df['lon'] >= lon_min) & (stations_df['lon'] <= lon_max)
                        )
                        nearby_stations = stations_df[mask]

                        print(f"  找到 {len(nearby_stations)} 个附近站点")

                        for _, station in nearby_stations.iterrows():
                            file_path = self.download_igs_whu_ztd(
                                station_id=station['station_id'],
                                date=fy3g_time
                            )
                            if file_path:
                                df = self.parse_igs_whu_file(file_path)
                                if not df.empty:
                                    df['lat'] = station['lat']
                                    df['lon'] = station['lon']
                                    df['height'] = station['height']
                                    df['source'] = 'igs_whu'
                                    all_gnss_data.append(df)

                elif source == 'euref':
                    # EUREF - 需要站点列表
                    if self.stations_file.exists():
                        stations_df = pd.read_csv(self.stations_file)
                        # 空间筛选欧洲站点
                        mask = (
                            (stations_df['lat'] >= lat_min) & (stations_df['lat'] <= lat_max) &
                            (stations_df['lon'] >= lon_min) & (stations_df['lon'] <= lon_max) &
                            (stations_df['lat'] >= 35) & (stations_df['lat'] <= 72) &  # 欧洲范围
                            (stations_df['lon'] >= -10) & (stations_df['lon'] <= 40)
                        )
                        nearby_stations = stations_df[mask]

                        print(f"  找到 {len(nearby_stations)} 个附近欧洲站点")

                        for _, station in nearby_stations.iterrows():
                            file_path = self.download_euref_ztd(
                                station_id=station['station_id'],
                                date=fy3g_time
                            )
                            if file_path:
                                df = self.parse_igs_whu_file(file_path)
                                if not df.empty:
                                    df['lat'] = station['lat']
                                    df['lon'] = station['lon']
                                    df['height'] = station['height']
                                    df['source'] = 'euref'
                                    all_gnss_data.append(df)

            except Exception as e:
                print(f"  错误: {source} 下载失败 - {e}")
                continue

        # 4. 合并所有数据
        if all_gnss_data:
            df_combined = pd.concat(all_gnss_data, ignore_index=True)

            # 时间筛选
            time_diff = np.abs((df_combined['datetime'] - fy3g_time).dt.total_seconds() / 3600)
            time_mask = time_diff <= time_window_hours
            df_final = df_combined[time_mask].copy()

            print(f"\n{'='*60}")
            print(f"合并结果:")
            print(f"  总数据: {len(df_combined)} 条")
            print(f"  时间筛选(±{time_window_hours}h): {len(df_final)} 条")
            print(f"  数据源分布:")
            for source in df_final['source'].unique():
                count = (df_final['source'] == source).sum()
                print(f"    {source}: {count} 条")
            print(f"{'='*60}")

            return df_final
        else:
            print("\n警告: 未下载到任何GNSS数据")
            return pd.DataFrame()


def main():
    """示例用法 - 展示多数据源和ERA5气象数据集成"""
    print("=" * 60)
    print("GNSS ZTD数据下载器 - 多数据源支持")
    print("=" * 60)



    # 下载并选择全球站点
    df_stations = downloader_ngl.select_global_uniform_stations(n_stations=100)
    print(f"选择了 {len(df_stations)} 个站点")

    # 处理单个站点示例（使用ERA5气象数据）
    if not df_stations.empty:
        test_station = df_stations.iloc[0]
        print(f"\n处理站点: {test_station['station_id']} "
              f"(Lat: {test_station['lat']:.2f}, Lon: {test_station['lon']:.2f})")

        df_pwv = downloader_ngl.process_station_to_pwv(
            station_id=test_station['station_id'],
            year=2023,
            station_lat=test_station['lat'],
            station_lon=test_station['lon'],
            station_height=test_station['height'],
            use_era5_meteo=True  # 使用ERA5气象数据
        )

        if not df_pwv.empty:
            print(f"成功处理 {len(df_pwv)} 条PWV数据")
            print(df_pwv.head())

    # ========== 示例2: SuomiNet数据源 (需要EarthScope认证) ==========
    print("\n[示例2] SuomiNet数据源 (EarthScope)")
    print("-" * 60)
    try:
        downloader_suominet = GNSSDownloader(data_source='suominet')

        # 下载单日数据
        test_date = datetime(2023, 11, 1)
        files = downloader_suominet.download_suominet_ztd(
            date=test_date,
            patterns=['b', 'c']
        )

        if files:
            print(f"下载了 {len(files)} 个文件")
            # 解析第一个文件
            df_suominet = downloader_suominet.parse_suominet_file(files[0])
            print(f"解析了 {len(df_suominet)} 条ZTD数据")
            print(df_suominet.head())

    except Exception as e:
        print(f"SuomiNet下载需要EarthScope认证: {e}")

    # ========== 示例3: IGS武汉大学数据源 ==========
    print("\n[示例3] IGS武汉大学数据源")
    print("-" * 60)
    downloader_igs = GNSSDownloader(data_source='igs_whu')

    # 下载示例站点（北京站）
    test_date = datetime(2023, 11, 1)
    igs_file = downloader_igs.download_igs_whu_ztd(
        station_id='bjfs',
        date=test_date
    )

    if igs_file:
        print(f"IGS武汉数据下载成功: {igs_file}")

    # ========== 示例4: 澳大利亚数据源 ==========
    print("\n[示例4] 澳大利亚对流层产品")
    print("-" * 60)
    downloader_aus = GNSSDownloader(data_source='australia')

    # 下载单日数据
    test_date = datetime(2023, 11, 1)
    aus_files = downloader_aus.download_australia_ztd(
        date=test_date,
        region='gar1'  # APREF + IGS
    )

    if aus_files:
        print(f"澳大利亚数据下载了 {len(aus_files)} 个文件")
        # 解析第一个文件
        df_aus = downloader_aus.parse_australia_file(aus_files[0])
        if not df_aus.empty:
            print(f"解析了 {len(df_aus)} 条数据")
            print(df_aus.head())

    # ========== 清理ERA5缓存 ==========
    downloader_ngl.era5_provider.close()

    print("\n" + "=" * 60)
    print("所有示例完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
