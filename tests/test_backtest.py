import numpy as np
import pandas as pd
import pytest

from stock_forecast.backtest import compute_metrics, walk_forward_validate
from stock_forecast.models import ForecastModel


def test_compute_metrics_matches_manual_calculation():
    actual = pd.Series([0.01, -0.02, 0.03, -0.01])
    predicted = pd.Series([0.01, -0.01, 0.02, 0.01])

    metrics = compute_metrics(actual, predicted)

    assert metrics["mae"] == pytest.approx(np.mean(np.abs(actual - predicted)))
    assert metrics["rmse"] == pytest.approx(np.sqrt(np.mean((actual - predicted) ** 2)))
    assert metrics["directional_accuracy"] == pytest.approx(0.75)  # 3 of 4 signs match
    assert metrics["n_predictions"] == 4


def test_walk_forward_uses_expanding_window_with_no_lookahead():
    n = 60
    dates = pd.bdate_range("2023-01-02", periods=n)
    df = pd.DataFrame(
        {"feature": np.arange(n, dtype=float), "Close": 100.0, "target_return": np.zeros(n)},
        index=dates,
    )

    train_sizes: list[int] = []

    class RecordingModel(ForecastModel):
        def fit(self, X, y):
            train_sizes.append(len(X))
            return self

        def predict(self, X):
            return np.zeros(len(X))

    predictions, metrics = walk_forward_validate(
        df,
        feature_cols=["feature"],
        target_col="target_return",
        model_factory=RecordingModel,
        min_train_size=20,
        retrain_every=10,
    )

    assert train_sizes == [20, 30, 40, 50]
    assert len(predictions) == n - 20
    assert metrics["n_predictions"] == n - 20


def test_walk_forward_perfect_model_scores_perfectly():
    n = 60
    dates = pd.bdate_range("2023-01-02", periods=n)
    target = pd.Series(np.linspace(-0.02, 0.02, n), index=dates)
    target = target.where(target != 0, 0.001)
    df = pd.DataFrame({"feature": target, "Close": 100.0, "target_return": target}, index=dates)

    class IdentityModel(ForecastModel):
        def fit(self, X, y):
            return self

        def predict(self, X):
            return X["feature"].to_numpy()

    _, metrics = walk_forward_validate(
        df,
        feature_cols=["feature"],
        target_col="target_return",
        model_factory=IdentityModel,
        min_train_size=20,
        retrain_every=10,
    )

    assert metrics["mae"] == pytest.approx(0.0, abs=1e-9)
    assert metrics["rmse"] == pytest.approx(0.0, abs=1e-9)
    assert metrics["directional_accuracy"] == pytest.approx(1.0)


def test_walk_forward_raises_when_not_enough_history():
    n = 5
    dates = pd.bdate_range("2023-01-02", periods=n)
    df = pd.DataFrame({"feature": np.arange(n, dtype=float), "Close": 100.0, "target_return": np.zeros(n)}, index=dates)

    with pytest.raises(ValueError):
        walk_forward_validate(
            df,
            feature_cols=["feature"],
            target_col="target_return",
            model_factory=lambda: None,
            min_train_size=100,
        )
