"""Model registry, keyed by name, all sharing the same fit/predict interface."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class ForecastModel(ABC):
    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> "ForecastModel":
        ...

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        ...


from stock_forecast.models.baseline import MovingAverageModel, NaiveLastValueModel  # noqa: E402
from stock_forecast.models.linear import LinearRegressionModel  # noqa: E402
from stock_forecast.models.random_forest import RandomForestModel  # noqa: E402

MODEL_REGISTRY: dict[str, type[ForecastModel]] = {
    "naive": NaiveLastValueModel,
    "moving_average": MovingAverageModel,
    "linear_regression": LinearRegressionModel,
    "random_forest": RandomForestModel,
}


def get_model(name: str, **params) -> ForecastModel:
    if name not in MODEL_REGISTRY:
        available = ", ".join(sorted(MODEL_REGISTRY))
        raise ValueError(f"Unknown model '{name}'. Available models: {available}")
    return MODEL_REGISTRY[name](**params)
