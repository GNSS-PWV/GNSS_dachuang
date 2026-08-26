# -*- coding: utf-8 -*-
"""
Saastamoinen 模型 —— 从气压/温度/水汽压/加权平均温度计算 PWV
由 SaastamoinenPWV.m 转换而来, 修正了原始 MATLAB 代码的两个 bug:
  1. 纬度双重弧度转换 (run脚本已转弧度, 函数内再deg2rad, 导致重力修正项f失效)
  2. 温度单位: 原代码假设摄氏度输入再+273.15, 但数据中TS已是开尔文

输入:
  lat   纬度 (度)
  h     高程 (m)
  P     地面气压 (hPa)
  e     水汽压 (hPa)
  T     地面温度 (K)  -- 注意: 原MATLAB期望摄氏度, 此处统一用开尔文
  Tm    加权平均温度 (K)

输出:
  dict: ZTD_mm, ZHD_mm, ZWD_mm, Pi, PWV_mm
"""
import numpy as np

# 物理常数 (与 MATLAB 原代码一致)
K2P = 22.1          # 大气折射常数, K/hPa
K3 = 3.739e5        # 大气折射常数, K^2/hPa
RV = 461.495        # 水汽气体常数, J/(kg*K)
RHO_W = 1000.0      # 液态水密度, kg/m^3


def saastamoinen_pwv(lat_deg, h_m, P_hpa, e_hpa, T_K, Tm_K):
    """
    Saastamoinen 模型计算 ZTD -> ZHD -> ZWD -> Pi -> PWV.

    参数:
      lat_deg : 纬度, 单位 度
      h_m     : 高程, 单位 m
      P_hpa   : 地面气压, 单位 hPa
      e_hpa   : 水汽压, 单位 hPa
      T_K     : 地面温度, 单位 K (若输入摄氏度请先 +273.15)
      Tm_K    : 加权平均温度, 单位 K

    返回:
      dict: ZTD_mm, ZHD_mm, ZWD_mm, Pi, PWV_mm
    """
    lat_rad = np.deg2rad(lat_deg)

    # 重力修正项 f (修正: 原代码在run脚本中已转弧度后函数内又deg2rad, 此处统一从度转换一次)
    f = 1.0 - 0.00266 * np.cos(2.0 * lat_rad) - 0.00000028 * h_m

    # Saastamoinen 天顶总延迟 ZTD (mm)
    ZTD_mm = 0.002277 * (P_hpa / f + (0.05 + 1255.0 / T_K) * e_hpa) * 1000.0

    # 天顶干延迟 ZHD (mm)
    ZHD_mm = 0.002277 * P_hpa / f * 1000.0

    # 天顶湿延迟 ZWD (mm)
    ZWD_mm = ZTD_mm - ZHD_mm

    # 转换因子 Pi (无量纲)
    Pi = 1e8 / (RHO_W * RV * (K3 / Tm_K + K2P))

    # PWV (mm)
    PWV_mm = Pi * ZWD_mm

    return {
        "ZTD_mm": ZTD_mm,
        "ZHD_mm": ZHD_mm,
        "ZWD_mm": ZWD_mm,
        "Pi": Pi,
        "PWV_mm": PWV_mm,
    }


def saastamoinen_pwv_batch(lat_arr, h_arr, P_arr, e_arr, T_arr, Tm_arr):
    """批量计算, 输入均为 numpy 数组, 返回 PWV 数组 (mm)."""
    lat_arr = np.asarray(lat_arr, dtype=np.float64)
    h_arr = np.asarray(h_arr, dtype=np.float64)
    P_arr = np.asarray(P_arr, dtype=np.float64)
    e_arr = np.asarray(e_arr, dtype=np.float64)
    T_arr = np.asarray(T_arr, dtype=np.float64)
    Tm_arr = np.asarray(Tm_arr, dtype=np.float64)

    lat_rad = np.deg2rad(lat_arr)
    f = 1.0 - 0.00266 * np.cos(2.0 * lat_rad) - 0.00000028 * h_arr

    ZTD_mm = 0.002277 * (P_arr / f + (0.05 + 1255.0 / T_arr) * e_arr) * 1000.0
    ZHD_mm = 0.002277 * P_arr / f * 1000.0
    ZWD_mm = ZTD_mm - ZHD_mm
    Pi = 1e8 / (RHO_W * RV * (K3 / Tm_arr + K2P))
    PWV_mm = Pi * ZWD_mm

    return PWV_mm


if __name__ == "__main__":
    # 快速验证: 用数据文件 AEM00041217 的第一行地表值
    # TS=288.23575 K, PS=1016.32 hPa, WPS=9.391 hPa, Tm=285.35 K, LAT=24.4333, ELV=16
    r = saastamoinen_pwv(24.4333, 16.0, 1016.32, 9.391, 288.23575, 285.3517)
    print("Saastamoinen 验证 (AEM00041217 首行):")
    for k, v in r.items():
        print(f"  {k:8s} = {v:.6f}")
    print(f"  数据中真值 PWV = 15.401196 mm")
    print(f"  误差 = {abs(r['PWV_mm'] - 15.401196):.6f} mm")
