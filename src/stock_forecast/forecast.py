"""Recursive one-day-ahead forecasting past the last known trading day."""

from __future__ import annotations

import numpy as np
import pandas as pd

from stock_forecast.features import FEATURE_COLUMNS, TARGET_COLUMN, build_features, compute_indicators
from stock_forecast.models import get_model


def forecast_future(raw: pd.DataFrame, model_name: str, model_params: dict, future_days: int) -> pd.DataFrame:
    """Predict the next `future_days` closing prices, one day at a time."""
    training_data = build_features(raw, horizon=1)
    model = get_model(model_name, **model_params)
    model.fit(training_data[FEATURE_COLUMNS], training_data[TARGET_COLUMN])

    working = raw[["Open", "High", "Low", "Close", "Volume"]].copy()
    future_dates = pd.bdate_range(working.index[-1] + pd.Timedelta(days=1), periods=future_days)

    rows = []
    for date in future_dates:
        latest_features = compute_indicators(working).iloc[[-1]][FEATURE_COLUMNS]
        predicted_return = float(model.predict(latest_features)[0])
        last_close = float(working["Close"].iloc[-1])
        predicted_close = last_close * (1 + predicted_return)

        rows.append({"date": date, "predicted_close": predicted_close, "predicted_return": predicted_return})

        working.loc[date] = {
            "Open": predicted_close,
            "High": predicted_close,
            "Low": predicted_close,
            "Close": predicted_close,
            "Volume": working["Volume"].iloc[-1],
        }

    return pd.DataFrame(rows).set_index("date")


def add_uncertainty_bounds(forecast: pd.DataFrame, daily_error: float) -> pd.DataFrame:
    """Widen the band around each predicted price by sqrt(days ahead)."""
    out = forecast.copy()
    days_ahead = np.arange(1, len(out) + 1)
    spread = out["predicted_close"] * daily_error * np.sqrt(days_ahead)
    out["lower_bound"] = out["predicted_close"] - spread
    out["upper_bound"] = out["predicted_close"] + spread
    return out
