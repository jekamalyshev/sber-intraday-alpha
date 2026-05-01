"""
src/features.py
===============
Feature engineering functions for 5-minute SBER OHLCV candle data.

All functions follow the same contract:
    - Accept a pd.DataFrame
    - Return a new pd.DataFrame (copy)
    - Never introduce look-ahead bias (only shift(>=1) or current-bar values)

Author: Evgeniy Malishev (jekamalyshev)
"""
from __future__ import annotations

from typing import Iterable, List

import numpy as np
import pandas as pd
from pandas import DataFrame, concat


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REQUIRED_OHLCV = {"OPEN", "HIGH", "LOW", "CLOSE", "VOL"}
EPS = 1e-12  # Guard against division by zero on flat candles


# ---------------------------------------------------------------------------
# 0. Supervised-learning framing (from machinelearningmastery.com)
# ---------------------------------------------------------------------------

def series_to_supervised(
    data: pd.DataFrame,
    n_in: int = 1,
    n_out: int = 1,
    dropnan: bool = True,
) -> pd.DataFrame:
    """
    Frame a multivariate time series as a supervised learning dataset.

    Parameters
    ----------
    data : pd.DataFrame
        Multivariate time series (columns = features, rows = timesteps).
    n_in : int
        Number of lag steps to use as input features.
    n_out : int
        Number of forward steps to use as output (usually 1 for single-step prediction).
    dropnan : bool
        Whether to drop rows that contain NaN after shifting.

    Returns
    -------
    pd.DataFrame
        Wide dataframe with lagged input columns and forward output columns.

    References
    ----------
    https://machinelearningmastery.com/convert-time-series-supervised-learning-problem-python/
    """
    n_vars = 1 if isinstance(data, list) else data.shape[1]
    df = DataFrame(data)
    col_list = data.columns.tolist()
    cols, names = [], []

    # Lagged input columns: t-n_in, ..., t-1
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [
            f"{col_list[j]}_var{j + 1}(t-{i})" for j in range(n_vars)
        ]

    # Forecast output columns: t, t+1, ..., t+n_out-1
    for i in range(n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [f"{col_list[j]}_var{j + 1}(t)" for j in range(n_vars)]
        else:
            names += [f"{col_list[j]}_var{j + 1}(t+{i})" for j in range(n_vars)]

    agg = concat(cols, axis=1)
    agg.columns = names
    if dropnan:
        agg.dropna(inplace=True)
    return agg


# ---------------------------------------------------------------------------
# 1. OHLCV preparation
# ---------------------------------------------------------------------------

def prepare_ohlcv_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate columns, cast numeric types, parse datetime, sort chronologically.

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataframe from Finam CSV export.

    Returns
    -------
    pd.DataFrame
        Cleaned and sorted dataframe.
    """
    missing = REQUIRED_OHLCV.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    out = df.copy(deep=True)

    # Cast OHLCV to float
    for col in ["OPEN", "HIGH", "LOW", "CLOSE", "VOL"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    # Parse datetime from DATE + TIME columns if available
    if {"DATE", "TIME"}.issubset(out.columns):
        date_str = out["DATE"].astype(str).str.strip()
        time_str = out["TIME"].astype(str).str.zfill(6).str.strip()
        out["datetime"] = pd.to_datetime(
            date_str + time_str,
            format="%Y%m%d%H%M%S",
            errors="coerce",
        )
        out = out.sort_values("datetime", kind="stable").reset_index(drop=True)

    return out


# ---------------------------------------------------------------------------
# 2. Candle geometry features
# ---------------------------------------------------------------------------

def add_domain_features_current_bar(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add candle geometry and single-bar return features.

    All features use only current-bar OHLCV + previous bar close/high/low.
    No look-ahead bias.
    """
    out = df.copy(deep=True)

    rng = out["HIGH"] - out["LOW"]
    safe_rng = rng.where(rng.abs() > EPS, np.nan)
    max_oc = np.maximum(out["OPEN"], out["CLOSE"])
    min_oc = np.minimum(out["OPEN"], out["CLOSE"])

    prev_close = out["CLOSE"].shift(1)
    prev_high = out["HIGH"].shift(1)
    prev_low = out["LOW"].shift(1)

    out["candle_body"] = out["CLOSE"] - out["OPEN"]
    out["body_abs"] = out["candle_body"].abs()
    out["candle_range"] = rng
    out["upper_wick"] = out["HIGH"] - max_oc
    out["lower_wick"] = min_oc - out["LOW"]

    out["body_to_range"] = (out["body_abs"] / safe_rng).clip(0, 1)
    out["upper_wick_to_range"] = (out["upper_wick"] / safe_rng).clip(0, 1)
    out["lower_wick_to_range"] = (out["lower_wick"] / safe_rng).clip(0, 1)
    out["close_pos_in_range"] = ((out["CLOSE"] - out["LOW"]) / safe_rng).clip(0, 1)
    out["open_pos_in_range"] = ((out["OPEN"] - out["LOW"]) / safe_rng).clip(0, 1)

    out["direction"] = np.sign(out["candle_body"]).astype("float64")
    out["is_green"] = (out["CLOSE"] > out["OPEN"]).astype("int8")
    out["is_red"] = (out["CLOSE"] < out["OPEN"]).astype("int8")
    out["is_doji_like"] = (out["body_to_range"] <= 0.1).astype("int8")

    out["ret_1"] = out["CLOSE"].pct_change(1)
    out["gap_from_prev_close"] = out["OPEN"] / prev_close - 1.0
    out["close_to_prev_high"] = out["CLOSE"] / prev_high - 1.0
    out["close_to_prev_low"] = out["CLOSE"] / prev_low - 1.0
    out["range_pct_close"] = out["candle_range"] / out["CLOSE"].replace(0, np.nan)
    out["money_flow_proxy"] = ((out["CLOSE"] - out["LOW"]) - (out["HIGH"] - out["CLOSE"])) / safe_rng * out["VOL"]
    out["body_x_vol"] = out["body_abs"] * out["VOL"]
    out["signed_vol"] = out["direction"] * out["VOL"]

    return out


# ---------------------------------------------------------------------------
# 3. Rolling context features
# ---------------------------------------------------------------------------

def add_rolling_context_features(
    df: pd.DataFrame,
    windows: Iterable[int] = (6, 12, 24),
) -> pd.DataFrame:
    """
    Add rolling mean, std, z-score for close, range, body and volume.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with candle geometry features already added.
    windows : Iterable[int]
        Rolling window sizes in bars. Default: 6=30m, 12=60m, 24=120m.
    """
    out = df.copy(deep=True)

    for w in windows:
        # --- Returns
        out[f"ret_{w}"] = out["CLOSE"].pct_change(w)

        # --- Range rolling stats
        rng_mean = out["candle_range"].rolling(w, min_periods=w).mean()
        rng_std = out["candle_range"].rolling(w, min_periods=w).std()
        out[f"range_mean_{w}"] = rng_mean
        out[f"range_std_{w}"] = rng_std
        out[f"range_zscore_{w}"] = (out["candle_range"] - rng_mean) / rng_std.replace(0, np.nan)

        # --- Body rolling stats
        out[f"body_abs_mean_{w}"] = out["body_abs"].rolling(w, min_periods=w).mean()

        # --- Close rolling stats
        close_ma = out["CLOSE"].rolling(w, min_periods=w).mean()
        close_std = out["CLOSE"].rolling(w, min_periods=w).std()
        out[f"close_ma_{w}"] = close_ma
        out[f"close_std_{w}"] = close_std
        out[f"close_vs_ma_{w}"] = out["CLOSE"] / close_ma - 1.0
        out[f"close_zscore_{w}"] = (out["CLOSE"] - close_ma) / close_std.replace(0, np.nan)

        # --- Volume rolling stats
        vol_ma = out["VOL"].rolling(w, min_periods=w).mean()
        out[f"vol_ma_{w}"] = vol_ma
        out[f"vol_std_{w}"] = out["VOL"].rolling(w, min_periods=w).std()
        out[f"vol_ratio_{w}"] = out["VOL"] / vol_ma.replace(0, np.nan)

    return out


# ---------------------------------------------------------------------------
# 4. Calendar / session features
# ---------------------------------------------------------------------------

def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add time-of-day and day-of-week features.
    Requires a 'datetime' column (added by prepare_ohlcv_dataframe).
    """
    if "datetime" not in df.columns:
        raise ValueError("Column 'datetime' not found. Run prepare_ohlcv_dataframe first.")

    out = df.copy(deep=True)
    dt = out["datetime"]

    out["hour"] = dt.dt.hour
    out["minute"] = dt.dt.minute
    out["dayofweek"] = dt.dt.dayofweek  # 0=Mon, 4=Fri
    out["weekofyear"] = dt.dt.isocalendar().week.astype(int)

    # Cyclic encodings
    out["hour_sin"] = np.sin(2 * np.pi * out["hour"] / 24)
    out["hour_cos"] = np.cos(2 * np.pi * out["hour"] / 24)
    out["minute_sin"] = np.sin(2 * np.pi * out["minute"] / 60)
    out["minute_cos"] = np.cos(2 * np.pi * out["minute"] / 60)
    out["dow_sin"] = np.sin(2 * np.pi * out["dayofweek"] / 5)
    out["dow_cos"] = np.cos(2 * np.pi * out["dayofweek"] / 5)

    # MOEX session: 10:00 – 18:40
    session_start_minutes = 10 * 60
    out["minutes_from_session_open"] = out["hour"] * 60 + out["minute"] - session_start_minutes
    out["is_opening_30m"] = (out["minutes_from_session_open"] < 30).astype("int8")
    out["is_first_hour"] = (out["minutes_from_session_open"] < 60).astype("int8")

    # Bar index within the day
    out["date_only"] = dt.dt.date
    out["bar_in_day"] = out.groupby("date_only").cumcount()
    out["bars_in_day"] = out.groupby("date_only")["bar_in_day"].transform("max") + 1
    out["is_first_bar_of_day"] = (out["bar_in_day"] == 0).astype("int8")
    out["is_last_bar_of_day"] = (out["bar_in_day"] == out["bars_in_day"] - 1).astype("int8")
    out = out.drop(columns=["date_only"])

    return out
