# -*- coding: utf-8 -*-
"""Prepare Mendeley Air Quality dataset (21 Slovenian stations) for the generic
non-uniform spatio-temporal forecasting framework.

Output: data/prepared.pkl
  records[sid] = {
     "feat": (T, D) float32 raw feature matrix (unscaled),
     "target": (T,) float32 PM2.5,
     "target_ok": (T,) bool (PM2.5 originally observed, not interpolated),
     "hour": (T,) int, "month": (T,) int,
     "coord": (3,) [lon, lat, elev_m],
     "times": (T,) datetime64
  }
"""
import os, glob
import numpy as np
import pandas as pd

ROOT = r"D:\gnss水汽反演\通用框架_CCFA"
DATA = os.path.join(ROOT, "data", "extracted")
META = os.path.join(ROOT, "data", "stations_meta.csv")
OUT = os.path.join(ROOT, "data", "prepared.pkl")

CLOUDS_ORDER = ["jasno", "delno oblačno", "pretežno oblačno", "oblačno"]
WIND_ORDER = ["S", "SV", "V", "JV", "J", "JZ", "Z", "SZ"]

NUM_COLS = ["PM10", "PM2.5", "temperature", "rain", "pressure", "precipitation", "wind_speed"]
FEAT_COLS = (["PM10", "temperature", "rain", "pressure", "precipitation", "wind_speed",
              "hour_sin", "hour_cos", "doy_sin", "doy_cos", "doy_sin2", "doy_cos2"]
             + [f"cloud_{i}" for i in range(len(CLOUDS_ORDER))] + ["cloud_na"]
             + [f"wd_{i}" for i in range(len(WIND_ORDER))] + ["wd_na"]
             + ["PM2.5_lag1"])  # target lag is appended below


def build_features(df):
    dt = pd.to_datetime(df["datetime"])
    hour = dt.dt.hour.values
    doy = dt.dt.dayofyear.values
    df = df.copy()
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
    df["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    df["doy_sin2"] = np.sin(4 * np.pi * doy / 365.25)
    df["doy_cos2"] = np.cos(4 * np.pi * doy / 365.25)
    for i, c in enumerate(CLOUDS_ORDER):
        df[f"cloud_{i}"] = (df["clouds"] == c).astype(np.float32)
    df["cloud_na"] = df["clouds"].isna().astype(np.float32)
    for i, w in enumerate(WIND_ORDER):
        df[f"wd_{i}"] = (df["wind_direction"] == w).astype(np.float32)
    df["wd_na"] = df["wind_direction"].isna().astype(np.float32)
    return df, hour, doy, dt


def main():
    meta = pd.read_csv(META).set_index("station")
    files = sorted(glob.glob(os.path.join(DATA, "E*.csv")))
    records = {}
    for f in files:
        sid = os.path.splitext(os.path.basename(f))[0]
        df = pd.read_csv(f)
        target_ok = df["PM2.5"].notna().values  # before filling
        df, hour, doy, dt = build_features(df)
        # PM2.5 lag 1 (shift within station, first row NaN -> fill with first valid)
        df["PM2.5_lag1"] = df["PM2.5"].shift(1)
        # interpolate numeric (linear, both directions), then ffill/bfill for residual gaps
        df[NUM_COLS + ["PM2.5_lag1"]] = df[NUM_COLS + ["PM2.5_lag1"]].interpolate(method="linear", limit_direction="both")
        df[NUM_COLS + ["PM2.5_lag1"]] = df[NUM_COLS + ["PM2.5_lag1"]].ffill().bfill()
        feat = df[FEAT_COLS].to_numpy(dtype=np.float32)
        target = df["PM2.5"].to_numpy(dtype=np.float32)
        assert len(feat) == len(target) == len(target_ok), sid
        coord = meta.loc[sid, ["lon", "lat", "elev_m"]].to_numpy(dtype=np.float64)
        records[sid] = {
            "feat": feat, "target": target, "target_ok": target_ok,
            "hour": hour.astype(np.int32), "month": (pd.to_datetime(dt).dt.month.values).astype(np.int32),
            "coord": coord, "times": dt.to_numpy(),
        }
        print(sid, feat.shape, "nan in feat:", np.isnan(feat).sum(), "target nan:", np.isnan(target).sum())
    pd.to_pickle(records, OUT)
    print("saved", OUT, "stations:", len(records))


if __name__ == "__main__":
    main()