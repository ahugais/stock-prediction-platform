"""Pulls historical price data from yfinance, with local caching."""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import yfinance as yf

from stock_forecast.config import RunConfig

REQUIRED_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]
CACHE_TTL_SECONDS = 60 * 60  # 1 hour


def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = df.columns.get_level_values(0)
    return df


def _download(ticker: str, start: str, end: str) -> pd.DataFrame:
    raw = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    if raw is None or raw.empty:
        raise ValueError(
            f"No price data returned for ticker '{ticker}' between {start} and {end}. "
            "Check the symbol and date range."
        )
    raw = _flatten_columns(raw)
    raw = raw[REQUIRED_COLUMNS].copy()
    raw.index.name = "Date"
    return raw


def fetch_price_history(ticker: str, start: str, end: str, refresh: bool = False) -> pd.DataFrame:
    """Return daily OHLCV history for `ticker`, cached under data/raw/ for CACHE_TTL_SECONDS."""
    cache_path = RunConfig(ticker=ticker, start=start, end=end).raw_path()

    if cache_path.exists() and not refresh:
        cache_age_seconds = time.time() - cache_path.stat().st_mtime
        if cache_age_seconds < CACHE_TTL_SECONDS:
            cached = pd.read_csv(cache_path, index_col="Date", parse_dates=True)
            cached_start, cached_end = pd.Timestamp(start), pd.Timestamp(end)
            if not cached.empty and cached.index.min() <= cached_start and cached.index.max() >= cached_end - pd.Timedelta(days=5):
                return cached.loc[(cached.index >= cached_start) & (cached.index <= cached_end)]

    df = _download(ticker, start, end)
    save_raw(df, cache_path)
    return df


def save_raw(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index_label="Date")


def load_raw(ticker: str) -> pd.DataFrame:
    path = RunConfig(ticker=ticker, start="", end="").raw_path()
    if not path.exists():
        raise FileNotFoundError(f"No cached raw data for '{ticker}' at {path}")
    return pd.read_csv(path, index_col="Date", parse_dates=True)
