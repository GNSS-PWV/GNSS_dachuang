# -*- coding: utf-8 -*-
"""Reconstruct a phase-2 compatible profile from parsed IGRA levels.

This is a pipeline-validation product, not a replacement for the teacher's
official labels.  It derives full-column PWV by hydrostatic integration of
IGRA vapor pressure and derives ZWD through the standard Tm conversion.  The
result must be kept separate from official training until independently
validated against the existing labels or ERA5.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

G = 9.80665
EPSILON = 0.622
K2P = 22.1
K3 = 3.739e5
RHO_W = 1000.0
RV = 461.495


def pi_from_tm(tm_k: float) -> float:
    return 1e8 / (RHO_W * RV * (K3 / tm_k + K2P))


def integrate_profile(group: pd.DataFrame, lat: float, lon: float, station_elv: float) -> dict[str, object] | None:
    g = group.sort_values("PRESSURE_HPA", ascending=False).copy()
    g = g.replace([np.inf, -np.inf], np.nan).dropna(subset=["PRESSURE_HPA", "ELV_M", "TEMP_K", "VAPOR_PRESSURE_HPA"])
    g = g[(g["PRESSURE_HPA"] > 0) & (g["VAPOR_PRESSURE_HPA"] >= 0) & (g["TEMP_K"] > 150)]
    if len(g) < 2:
        return None
    p = g["PRESSURE_HPA"].to_numpy(float) * 100.0
    e = g["VAPOR_PRESSURE_HPA"].to_numpy(float) * 100.0
    t = g["TEMP_K"].to_numpy(float)
    q = EPSILON * e / np.maximum(p - (1.0 - EPSILON) * e, 1.0)
    # Pressure decreases upward.  Trapezoid integration gives kg/m2 ~= mm.
    trapz = getattr(np, "trapezoid", np.trapz)
    pwv = float(trapz(q, p * -1.0) / G)
    if not np.isfinite(pwv) or pwv <= 0:
        return None
    numerator = float(trapz(e / t, p * -1.0))
    denominator = float(trapz(e / (t * t), p * -1.0))
    tm = numerator / denominator if denominator > 0 else np.nan
    if not np.isfinite(tm) or tm < 150 or tm > 350:
        return None
    zwd = pwv / pi_from_tm(tm)
    stamp = pd.Timestamp(g["TIME"].iloc[0])
    zhd = 2.2768 * g["PRESSURE_HPA"].to_numpy(float) / (1 - 0.00266 * np.cos(np.deg2rad(2 * lat)) - 0.00028 * g["ELV_M"].to_numpy(float) / 1000.0)
    levels = g[["ELV_M", "TEMP_K", "PRESSURE_HPA", "VAPOR_PRESSURE_HPA"]].to_numpy(float)
    return {
        "stamp": stamp, "pwv": pwv, "zwd": zwd, "tm": tm,
        "levels": levels, "zhd": zhd, "lat": lat, "lon": lon,
        "station_elv": station_elv,
    }


def convert(input_csv: str | Path, output_txt: str | Path, lat: float, lon: float, station_elv: float, years: set[int]) -> dict[str, int]:
    df = pd.read_csv(input_csv)
    df["TIME"] = pd.to_datetime(df["TIME"], errors="coerce")
    df = df[df["TIME"].dt.year.isin(years)].copy()
    station = str(df["station_id"].iloc[0]) if not df.empty else Path(input_csv).stem.split("_levels")[0]
    rows: list[dict[str, object]] = []
    accepted = 0
    for _, group in df.groupby("TIME", sort=True):
        item = integrate_profile(group, lat, lon, station_elv)
        if item is None:
            continue
        accepted += 1
        stamp = item["stamp"]
        for i, level in enumerate(item["levels"]):
            elv, ts, ps, wps = level
            rows.append({
                "TIME": stamp, "YEAR": stamp.year, "DOY": stamp.dayofyear,
                "LAT": lat, "LON": lon, "ELV": elv, "TS": ts, "PS": ps,
                "WPS": wps, "ZWD": item["zwd"], "ZHD": item["zhd"][i],
                "ZTD": item["zwd"] + item["zhd"][i], "PWV": item["pwv"], "Tm": item["tm"],
            })
    out = Path(output_txt); out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=["TIME", "YEAR", "DOY", "LAT", "LON", "ELV", "TS", "PS", "WPS", "ZWD", "ZHD", "ZTD", "PWV", "Tm"]).to_csv(out, index=False, float_format="%.8f")
    return {"station_id": station, "accepted_profiles": accepted, "level_rows": len(rows), "years": sorted(years), "label_method": "IGRA hydrostatic PWV + physical Tm-to-ZWD reconstruction; experimental"}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-csv", required=True)
    p.add_argument("--output-txt", required=True)
    p.add_argument("--lat", type=float, required=True)
    p.add_argument("--lon", type=float, required=True)
    p.add_argument("--station-elv", type=float, default=0.0)
    p.add_argument("--years", default="2014,2015,2016,2017,2018,2019")
    args = p.parse_args()
    print(convert(args.input_csv, args.output_txt, args.lat, args.lon, args.station_elv, {int(x) for x in args.years.split(",")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
