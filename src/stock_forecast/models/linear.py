from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from stock_forecast.models import ForecastModel


class LinearRegressionModel(ForecastModel):
    def __init__(self, **kwargs):
        self._model = LinearRegression(**kwargs)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "LinearRegressionModel":
        self._model.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self._model.predict(X)
