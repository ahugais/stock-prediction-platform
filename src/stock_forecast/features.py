"""Technical indicators and the leakage-free return target."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from stock_forecast.config import RunConfig

TARGET_COLUMN = "target_return"

FEATURE_COLUMNS = [
    "daily_return",
    "log_return",
    "sma_5",
    "sma_10",
    "sma_20",
    "ema_12",
    "rsi_14",
    "volatility_10",
    "bb_upper_20",
    "bb_lower_20",
    "bb_pct_b",
    "lag_return_1",
    "lag_return_2",
    "lag_return_3",
]


def _rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.where(avg_loss != 0, 100.0)


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Compute FEATURE_COLUMNS from raw OHLCV data, no target, no dropna."""
    out = df.copy()
    close = out["Close"]

    out["daily_return"] = close.pct_change()
    out["log_return"] = np.log(close / close.shift(1))

    out["sma_5"] = close.rolling(5).mean()
    out["sma_10"] = close.rolling(10).mean()
    out["sma_20"] = close.rolling(20).mean()
    out["ema_12"] = close.ewm(span=12, adjust=False).mean()

    out["rsi_14"] = _rsi(close, 14)

    out["volatility_10"] = out["daily_return"].rolling(10).std()

    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    out["bb_upper_20"] = bb_mid + 2 * bb_std
    out["bb_lower_20"] = bb_mid - 2 * bb_std
    out["bb_pct_b"] = (close - out["bb_lower_20"]) / (out["bb_upper_20"] - out["bb_lower_20"])

    out["lag_return_1"] = out["daily_return"].shift(1)
    out["lag_return_2"] = out["daily_return"].shift(2)
    out["lag_return_3"] = out["daily_return"].shift(3)

    return out


def build_features(df: pd.DataFrame, horizon: int = 1) -> pd.DataFrame:
    """Add indicators plus a next-`horizon`-day return target, drop NaN rows."""
    out = compute_indicators(df)
    out[TARGET_COLUMN] = out["Close"].shift(-horizon) / out["Close"] - 1
    out = out.dropna(subset=FEATURE_COLUMNS + [TARGET_COLUMN])
    return out


def save_features(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index_label="Date")


def load_features(ticker: str) -> pd.DataFrame:
    path = RunConfig(ticker=ticker, start="", end="").processed_path()
    if not path.exists():
        raise FileNotFoundError(f"No cached processed features for '{ticker}' at {path}")
    return pd.read_csv(path, index_col="Date", parse_dates=True)
