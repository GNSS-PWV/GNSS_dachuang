# -*- coding: utf-8 -*-
"""
GPT3 经验对流层模型 (Python 移植版)
====================================
由服务器上的 MATLAB 版 gpt3_1_fast.m / gpt3_1_fast_readGrid.m / doy2mjd.m 移植而来
(GPT3: Landskron & Boehm 2018, VMF3/GPT3, J Geod 92:349)

用途: 在任意位置/时刻计算
  p   : 气压     (hPa)
  T   : 温度     (摄氏度)
  Tm  : 加权平均温度 (Kelvin)   <- 本项目 GPT3 基线需要
  e   : 水汽压   (hPa)

核心: 1°x1° 全球网格 gpt3_1.grd (a0/A1/B1/A2/B2 年/半年谐波系数) +
      双线性插值 + 高度归算(温度直减率、气压指数衰减、水汽压 Askne-Nordius)。

用法:
  from gpt3 import GPT3
  gpt3 = GPT3('gpt3_1.grd')
  p, T, Tm, e = gpt3.compute(lat_deg, lon_deg, h_ell_m, year, doy, hour)
"""
import os
import numpy as np


class GPT3:
    def __init__(self, grd_path):
        # 读取网格: 跳过 '%' 开头的表头
        C = np.loadtxt(grd_path, comments='%')
        if C.shape[1] != 64:
            raise ValueError('grd 列数应为 64, 实际 %d' % C.shape[1])
        # 1-based 列 -> 0-based
        self.p_grid = C[:, 2:7]          # 气压 Pa (a0,A1,B1,A2,B2)
        self.T_grid = C[:, 7:12]         # 温度 K
        self.Q_grid = C[:, 12:17] / 1000.0   # 比湿 kg/kg
        self.dT_grid = C[:, 17:22] / 1000.0  # 温度直减率 K/m
        self.u_grid = C[:, 22]           # 大地水准面起伏 m
        self.Hs_grid = C[:, 23]          # 正高网格高度 m
        self.ah_grid = C[:, 24:29] / 1000.0
        self.aw_grid = C[:, 29:34] / 1000.0
        self.la_grid = C[:, 34:39]       # 水汽衰减因子
        self.Tm_grid = C[:, 39:44]       # 加权平均温度 K
        self.Gn_h_grid = C[:, 44:49] / 100000.0
        self.Ge_h_grid = C[:, 49:54] / 100000.0
        self.Gn_w_grid = C[:, 54:59] / 100000.0
        self.Ge_w_grid = C[:, 59:64] / 100000.0
        self.nrow = C.shape[0]           # 应为 180*360

    # ---- 核心: 与 MATLAB gpt3_1_fast 一一对应, 输入为弧度/米/mjd ----
    def _core(self, mjd, lat_rad, lon_rad, h_ell, it):
        lat = np.atleast_1d(np.asarray(lat_rad, dtype=np.float64))
        lon = np.atleast_1d(np.asarray(lon_rad, dtype=np.float64))
        h = np.atleast_1d(np.asarray(h_ell, dtype=np.float64))
        n = len(lat)

        # mjd -> doy (移植 MATLAB 的儒略日转公历/年积日算法)
        hour = np.floor((mjd - np.floor(mjd)) * 24)
        minute = np.floor((((mjd - np.floor(mjd)) * 24) - hour) * 60)
        sec = ((((mjd - np.floor(mjd)) * 24) - hour) * 60 - minute) * 60
        minute = minute + (sec == 60).astype(int)
        sec[sec == 60] = 0
        hour = hour + (minute == 60).astype(int)
        minute[minute == 60] = 0
        jd = mjd + 2400000.5
        jd[hour == 24] += 1
        hour[hour == 24] = 0

        jd_int = np.floor(jd + 0.5)
        aa = jd_int + 32044
        bb = np.floor((4 * aa + 3) / 146097.0)
        cc = aa - np.floor((bb * 146097) / 4.0)
        dd = np.floor((4 * cc + 3) / 1461.0)
        ee = cc - np.floor((1461 * dd) / 4.0)
        mm = np.floor((5 * ee + 2) / 153.0)
        day = ee - np.floor((153 * mm + 2) / 5.0) + 1
        month = mm + 3 - 12 * np.floor(mm / 10.0)
        year = bb * 100 + dd - 4800 + np.floor(mm / 10.0)

        leap = ((np.mod(year, 4) == 0) & (np.mod(year, 100) != 0)) | (np.mod(year, 400) == 0)
        days = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
        m_idx = (month - 1).astype(int)
        doy = np.zeros(n)
        for k in range(n):
            doy[k] = days[:m_idx[k]].sum() + day[k]
        doy = doy + (leap & (month > 2)).astype(int)
        doy = doy + (mjd - np.floor(mjd))   # 加小数部分

        if it == 1:
            cosfy = np.zeros(n); coshy = np.zeros(n)
            sinfy = np.zeros(n); sinhy = np.zeros(n)
        else:
            doy_ang = doy / 365.25 * 2 * np.pi
            cosfy = np.cos(doy_ang)
            coshy = np.cos(doy_ang * 2)
            sinfy = np.sin(doy_ang)
            sinhy = np.sin(doy_ang * 2)

        # 网格索引
        plon = np.where(lon < 0, (lon + 2 * np.pi) * 180 / np.pi, lon * 180 / np.pi)
        ppod = (-lat + np.pi / 2) * 180 / np.pi
        ipod = np.floor(ppod + 1).astype(int)
        ilon = np.floor(plon + 1).astype(int)
        diffpod = ppod - (ipod - 0.5)
        difflon = plon - (ilon - 0.5)
        ipod[ipod == 181] = 180
        ilon[ilon == 361] = 1
        ilon[ilon == 0] = 360
        bilinear = (ppod > 0.5) & (ppod < 179.5)

        # 谐波组合函数
        def harm(g):
            return (g[:, 0] + g[:, 1] * cosfy + g[:, 2] * sinfy +
                    g[:, 3] * coshy + g[:, 4] * sinhy)

        gm = 9.80665
        dMtr = 28.965e-3
        Rg = 8.3143

        p = np.zeros(n); T = np.zeros(n); Tm = np.zeros(n); e = np.zeros(n)

        # ---- 最近邻 (极区) ----
        nn = ~bilinear
        if nn.any():
            ix = (ipod[nn] - 1) * 360 + ilon[nn] - 1   # 0-based
            undu = self.u_grid[ix]
            hgt = h[nn] - undu
            T0 = harm(self.T_grid[ix])
            p0 = harm(self.p_grid[ix])
            Q = harm(self.Q_grid[ix])
            dT = harm(self.dT_grid[ix])
            redh = hgt - self.Hs_grid[ix]
            T[nn] = T0 + dT * redh - 273.15
            Tv = T0 * (1 + 0.6077 * Q)
            c = gm * dMtr / (Rg * Tv)
            p[nn] = p0 * np.exp(-c * redh) / 100.0
            la = harm(self.la_grid[ix])
            Tm[nn] = harm(self.Tm_grid[ix])
            e0 = Q * p0 / (0.622 + 0.378 * Q) / 100.0
            # 注意: p[nn] 已是 hPa, p0 是 Pa -> 100*p/p0 无量纲
            e[nn] = e0 * (100.0 * p[nn] / p0) ** (la + 1)

        # ---- 双线性 ----
        bl = bilinear
        if bl.any():
            ipod1 = ipod[bl] + np.sign(diffpod[bl]).astype(int)
            ilon1 = ilon[bl] + np.sign(difflon[bl]).astype(int)
            ilon1[ilon1 == 361] = 1
            ilon1[ilon1 == 0] = 360
            i0 = (ipod[bl] - 1) * 360 + ilon[bl] - 1
            i1 = (ipod1 - 1) * 360 + ilon[bl] - 1
            i2 = (ipod[bl] - 1) * 360 + ilon1 - 1
            i3 = (ipod1 - 1) * 360 + ilon1 - 1
            idx = np.stack([i0, i1, i2, i3], axis=1)   # (m,4)

            def gather(g):
                return g[idx]                           # (m,4,5) -> harmonic
            def harm4(g):
                g = gather(g)
                return (g[:, :, 0] + g[:, :, 1] * cosfy[bl][:, None] +
                        g[:, :, 2] * sinfy[bl][:, None] +
                        g[:, :, 3] * coshy[bl][:, None] +
                        g[:, :, 4] * sinhy[bl][:, None])   # (m,4)

            undul = self.u_grid[idx]
            hgt = h[bl][:, None] - undul
            T0 = harm4(self.T_grid)
            p0 = harm4(self.p_grid)
            Ql = harm4(self.Q_grid)
            dTl = harm4(self.dT_grid)
            Hs1 = self.Hs_grid[idx]
            redh = hgt - Hs1
            Tl = T0 + dTl * redh - 273.15
            Tv = T0 * (1 + 0.6077 * Ql)
            c = gm * dMtr / (Rg * Tv)
            pl = p0 * np.exp(-c * redh) / 100.0
            lal = harm4(self.la_grid)
            Tml = harm4(self.Tm_grid)
            e0 = Ql * p0 / (0.622 + 0.378 * Ql) / 100.0
            el = e0 * (100.0 * pl / p0) ** (lal + 1)

            dnpod1 = np.abs(diffpod[bl])
            dnpod2 = 1 - dnpod1
            dnlon1 = np.abs(difflon[bl])
            dnlon2 = 1 - dnlon1

            def bilin(v):
                R1 = dnpod2 * v[:, 0] + dnpod1 * v[:, 1]
                R2 = dnpod2 * v[:, 2] + dnpod1 * v[:, 3]
                return dnlon2 * R1 + dnlon1 * R2

            p[bl] = bilin(pl)
            T[bl] = bilin(Tl)
            e[bl] = bilin(el)
            Tm[bl] = bilin(Tml)

        return p, T, Tm, e

    # ---- 便捷接口: 输入 度/m + (year, doy, hour) ----
    def compute(self, lat_deg, lon_deg, h_ell_m, year, doy, hour=0.0, it=0):
        lat_deg = np.atleast_1d(np.asarray(lat_deg, dtype=np.float64))
        lon_deg = np.atleast_1d(np.asarray(lon_deg, dtype=np.float64))
        h = np.atleast_1d(np.asarray(h_ell_m, dtype=np.float64))
        year = np.atleast_1d(np.asarray(year, dtype=np.int64))
        doy = np.atleast_1d(np.asarray(doy, dtype=np.float64))
        hour = np.atleast_1d(np.asarray(hour, dtype=np.float64))
        n = len(lat_deg)
        mjd = np.zeros(n)
        for k in range(n):
            mjd[k] = self.doy2mjd(year[k], doy[k]) + hour[k] / 24.0
        lat_rad = np.deg2rad(lat_deg)
        lon_rad = np.deg2rad(lon_deg)
        return self._core(mjd, lat_rad, lon_rad, h, it)

    @staticmethod
    def doy2mjd(year, doy):
        """year + doy(1-based) -> MJD (与 doy2mjd.m 一致, 时刻为 00:00)"""
        import datetime as dt
        base = dt.date(int(year), 1, 1)
        d = base + dt.timedelta(days=int(doy) - 1)
        # 儒略日 = 公历转儒略日(标准算法)
        y, m, dd = d.year, d.month, d.day
        a = (14 - m) // 12
        yy = y + 4800 - a
        mm = m + 12 * a - 3
        jdn = dd + (153 * mm + 2) // 5 + 365 * yy + yy // 4 - yy // 100 + yy // 400 - 32045
        # jdn 为"正午"整数儒略日; 午夜(00:00)的 MJD = jdn - 2400001.0
        # (与 MATLAB juliandate(00:00) - 2400000.5 一致)
        return float(jdn) - 2400001.0


if __name__ == '__main__':
    import os
    grd = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'gpt3_1.grd')
    g = GPT3(grd)
    # 验证: 阿布扎比 AEM00041217 (2014-01-01, 00:00)
    p, T, Tm, e = g.compute(24.4333, 54.65, 16.0, 2014, 1, 0.0)
    print('AEM00041217 2014-01-01 00:00:')
    print('  p  = %.3f hPa' % p[0])
    print('  T  = %.3f C' % T[0])
    print('  Tm = %.3f K' % Tm[0])
    print('  e  = %.3f hPa' % e[0])
    print('(探空实测: PS=1016.32 hPa, TS=288.24K(15.09C), Tm=285.35K, WPS=9.39 hPa)')
