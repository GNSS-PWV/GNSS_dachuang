"""Past-only feature construction for phase-one forecasting.

The historical phase-one scripts contain useful experiments, but several of
their derived features use the value at the prediction timestamp.  This
module is an isolated replacement for future retraining.  It intentionally
uses only static metadata, deterministic calendar values, and lagged dynamic
observations from the same station and height.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


DYNAMIC_COLUMNS = ("PS", "WPS", "TS", "Tm")
LAGS = (1, 2, 3, 6, 12)
ROLLING_WINDOWS = (3, 6, 12)
GROUP_COLUMNS = ("station_id", "ELV")


def _groups(df: pd.DataFrame) -> list[str]:
    present = [column for column in GROUP_COLUMNS if column in df.columns]
    return present or ["__all__"]


def _grouped(df: pd.DataFrame, column: str, groups: list[str]):
    return df.groupby(groups, sort=False, dropna=False)[column]


def _calendar_features(data: pd.DataFrame) -> None:
    if "TIME" not in data.columns:
        return
    time = pd.to_datetime(data["TIME"], errors="coerce")
    data["hour"] = time.dt.hour
    data["month"] = time.dt.month
    data["day"] = time.dt.day
    data["weekday"] = time.dt.weekday
    data["season"] = ((data["month"] - 1) // 3).astype("float64")
    data["hour_sin"] = np.sin(2 * np.pi * data["hour"] / 24.0)
    data["hour_cos"] = np.cos(2 * np.pi * data["hour"] / 24.0)
    data["month_sin"] = np.sin(2 * np.pi * data["month"] / 12.0)
    data["month_cos"] = np.cos(2 * np.pi * data["month"] / 12.0)
    data["day_sin"] = np.sin(2 * np.pi * data["day"] / 31.0)
    data["day_cos"] = np.cos(2 * np.pi * data["day"] / 31.0)
    data["weekday_sin"] = np.sin(2 * np.pi * data["weekday"] / 7.0)
    data["weekday_cos"] = np.cos(2 * np.pi * data["weekday"] / 7.0)


def build_causal_features(
    frame: pd.DataFrame,
    *,
    target: str,
    sort: bool = True,
) -> tuple[pd.DataFrame, list[str]]:
    """Return a copy with strict past-only features for ``target``.

    ``target`` remains in the returned frame as the prediction label, but no
    current value of any dynamic column is used as an input feature.  This is
    deliberately stricter than allowing current TS/PS/WPS as cross-inputs;
    deployment can therefore use the same feature contract for every target.
    """
    if target not in DYNAMIC_COLUMNS:
        raise ValueError(f"target must be one of {DYNAMIC_COLUMNS}, got {target!r}")
    data = frame.copy()
    if "station_id" not in data.columns:
        data["station_id"] = "__single_station__"
    if "ELV" not in data.columns:
        data["ELV"] = 0.0
    if sort and "TIME" in data.columns:
        data = data.sort_values(["station_id", "ELV", "TIME"], kind="stable").reset_index(drop=True)

    _calendar_features(data)
    groups = _groups(data)
    feature_columns = [
        column
        for column in (
            "LAT", "LON", "ELV", "hour", "month", "day", "weekday", "season",
            "hour_sin", "hour_cos", "month_sin", "month_cos", "day_sin", "day_cos",
            "weekday_sin", "weekday_cos", "lat_sin", "lat_cos", "lon_sin", "lon_cos",
            "distance_from_equator",
        )
        if column in data.columns
    ]

    if "LAT" in data.columns:
        data["lat_sin"] = np.sin(np.deg2rad(data["LAT"]))
        data["lat_cos"] = np.cos(np.deg2rad(data["LAT"]))
        data["distance_from_equator"] = data["LAT"].abs()
    if "LON" in data.columns:
        data["lon_sin"] = np.sin(np.deg2rad(data["LON"]))
        data["lon_cos"] = np.cos(np.deg2rad(data["LON"]))
    feature_columns = [
        column
        for column in (
            "LAT", "LON", "ELV", "hour", "month", "day", "weekday", "season",
            "hour_sin", "hour_cos", "month_sin", "month_cos", "day_sin", "day_cos",
            "weekday_sin", "weekday_cos", "lat_sin", "lat_cos", "lon_sin", "lon_cos",
            "distance_from_equator",
        )
        if column in data.columns
    ]

    for source in DYNAMIC_COLUMNS:
        if source not in data.columns:
            continue
        grouped = _grouped(data, source, groups)
        for lag in LAGS:
            name = f"{source}_lag{lag}"
            data[name] = grouped.shift(lag)
            feature_columns.append(name)
        data[f"{source}_diff1"] = data[f"{source}_lag1"] - data[f"{source}_lag2"]
        data[f"{source}_diff2"] = data[f"{source}_lag2"] - data[f"{source}_lag3"]
        feature_columns.extend((f"{source}_diff1", f"{source}_diff2"))
        for window in ROLLING_WINDOWS:
            name = f"{source}_ma{window}"
            data[name] = grouped.transform(
                lambda values: values.shift(1).rolling(window, min_periods=1).mean()
            )
            feature_columns.append(name)
        for window in (3, 6):
            name = f"{source}_std{window}"
            data[name] = grouped.transform(
                lambda values: values.shift(1).rolling(window, min_periods=1).std()
            )
            feature_columns.append(name)

    # Cross-variable features also use the same historical timestamp.
    for left, right in (("PS", "TS"), ("WPS", "TS"), ("WPS", "PS"), ("TS", "Tm")):
        if f"{left}_lag1" not in data or f"{right}_lag1" not in data:
            continue
        ratio = f"{left}_{right}_ratio"
        product = f"{left}_{right}_product"
        difference = f"{left}_{right}_diff"
        data[ratio] = data[f"{left}_lag1"] / (data[f"{right}_lag1"] + 1e-8)
        data[product] = data[f"{left}_lag1"] * data[f"{right}_lag1"]
        data[difference] = data[f"{left}_lag1"] - data[f"{right}_lag1"]
        feature_columns.extend((ratio, product, difference))

    # De-duplicate while preserving stable feature order.
    feature_columns = list(dict.fromkeys(column for column in feature_columns if column in data.columns))
    if target in feature_columns:
        raise AssertionError(f"current target leaked into feature list: {target}")
    return data, feature_columns


def make_causal_sequences(
    frame: pd.DataFrame,
    *,
    target: str,
    time_steps: int = 24,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, list[str]]:
    """Build fixed-length sequences and target-time metadata."""
    if time_steps < 1:
        raise ValueError("time_steps must be positive")
    data, features = build_causal_features(frame, target=target)
    groups = _groups(data)
    usable = data.dropna(subset=features + [target])
    x_values: list[np.ndarray] = []
    y_values: list[float] = []
    metadata: list[dict[str, object]] = []
    for _, group in usable.groupby(groups, sort=False, dropna=False):
        group = group.sort_values("TIME", kind="stable") if "TIME" in group else group
        if len(group) <= time_steps:
            continue
        values = group[features].to_numpy(dtype=np.float32)
        labels = group[target].to_numpy(dtype=np.float32)
        for end in range(time_steps, len(group)):
            x_values.append(values[end - time_steps:end])
            y_values.append(float(labels[end]))
            row = group.iloc[end]
            metadata.append({
                "TIME": row.get("TIME"),
                "station_id": row.get("station_id"),
                "ELV": row.get("ELV"),
                "YEAR": row.get("YEAR"),
                "DOY": row.get("DOY"),
            })
    if not x_values:
        raise ValueError("no causal sequences were created")
    return np.stack(x_values), np.asarray(y_values, dtype=np.float32)[:, None], pd.DataFrame(metadata), features


def feature_contract(target: str) -> dict[str, object]:
    """Machine-readable contract used in reports and future training jobs."""
    if target not in DYNAMIC_COLUMNS:
        raise ValueError(f"target must be one of {DYNAMIC_COLUMNS}, got {target!r}")
    return {
        "target": target,
        "dynamic_current_values_allowed": False,
        "allowed_temporal_sources": "t-1 and earlier within station_id + ELV",
        "lags": list(LAGS),
        "rolling_windows": list(ROLLING_WINDOWS),
        "rolling_alignment": "shift(1) before rolling",
        "static_sources": ["LAT", "LON", "ELV"],
        "calendar_sources": ["TIME"],
    }
