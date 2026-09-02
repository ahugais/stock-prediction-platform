"""Expanding-window walk-forward validation."""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from stock_forecast.models import ForecastModel


def compute_metrics(actual: pd.Series, predicted: pd.Series) -> dict[str, float]:
    actual_arr = np.asarray(actual, dtype=float)
    predicted_arr = np.asarray(predicted, dtype=float)
    errors = actual_arr - predicted_arr
    return {
        "mae": float(np.mean(np.abs(errors))),
        "rmse": float(np.sqrt(np.mean(errors**2))),
        "directional_accuracy": float(np.mean(np.sign(actual_arr) == np.sign(predicted_arr))),
        "n_predictions": int(len(actual_arr)),
    }


def walk_forward_validate(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    model_factory: Callable[[], ForecastModel],
    min_train_size: int = 252,
    retrain_every: int = 21,
) -> tuple[pd.DataFrame, dict[str, float]]:
    df = df.sort_index()
    n = len(df)

    effective_min_train = min(min_train_size, max(int(n * 0.6), 10))
    if effective_min_train >= n:
        raise ValueError(
            f"Not enough rows ({n}) to walk-forward validate with min_train_size={min_train_size}. "
            "Choose a longer date range or a smaller --min-train-size."
        )

    records: list[dict] = []
    start = effective_min_train
    while start < n:
        end = min(start + retrain_every, n)
        train, test = df.iloc[:start], df.iloc[start:end]

        model = model_factory()
        model.fit(train[feature_cols], train[target_col])
        predicted_returns = model.predict(test[feature_cols])

        for date, actual_return, predicted_return, last_close in zip(
            test.index, test[target_col].to_numpy(), predicted_returns, test["Close"].to_numpy()
        ):
            records.append(
                {
                    "date": date,
                    "actual_return": actual_return,
                    "predicted_return": predicted_return,
                    "actual_close": last_close * (1 + actual_return),
                    "predicted_close": last_close * (1 + predicted_return),
                }
            )
        start = end

    predictions = pd.DataFrame.from_records(records).set_index("date")
    metrics = compute_metrics(predictions["actual_return"], predictions["predicted_return"])
    return predictions, metrics
