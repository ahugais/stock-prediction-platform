from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from stock_forecast.models import ForecastModel


class RandomForestModel(ForecastModel):
    def __init__(self, n_estimators: int = 200, max_depth: int | None = None, random_state: int = 42, **kwargs):
        self._model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1,
            **kwargs,
        )

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "RandomForestModel":
        self._model.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self._model.predict(X)
