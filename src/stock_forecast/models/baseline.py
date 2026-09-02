from __future__ import annotations

import numpy as np
import pandas as pd

from stock_forecast.models import ForecastModel


class NaiveLastValueModel(ForecastModel):
    """Predicts zero return, i.e. 'tomorrow's price equals today's'."""

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "NaiveLastValueModel":
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.zeros(len(X))


class MovingAverageModel(ForecastModel):
    """Predicts the mean of the last `window` observed training returns."""

    def __init__(self, window: int = 5):
        self.window = window
        self._mean = 0.0

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "MovingAverageModel":
        self._mean = float(pd.Series(y).tail(self.window).mean())
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.full(len(X), self._mean)
