import numpy as np
import pandas as pd

from stock_forecast.forecast import add_uncertainty_bounds, forecast_future


def _synthetic_ohlcv(n: int = 100, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    returns = rng.normal(loc=0.0005, scale=0.01, size=n)
    close = 100 * np.cumprod(1 + returns)
    dates = pd.bdate_range("2023-01-02", periods=n)
    return pd.DataFrame(
        {
            "Open": close,
            "High": close * 1.01,
            "Low": close * 0.99,
            "Close": close,
            "Volume": np.full(n, 1_000_000),
        },
        index=dates,
    )


def test_forecast_future_returns_requested_number_of_future_business_days():
    raw = _synthetic_ohlcv()
    forecast = forecast_future(raw, "naive", {}, future_days=5)

    assert len(forecast) == 5
    assert forecast.index[0] > raw.index[-1]
    assert forecast.index.is_monotonic_increasing
    assert forecast.index.weekday.max() <= 4


def test_forecast_future_naive_model_predicts_a_flat_line():
    raw = _synthetic_ohlcv()
    forecast = forecast_future(raw, "naive", {}, future_days=5)

    last_close = float(raw["Close"].iloc[-1])
    np.testing.assert_allclose(forecast["predicted_close"].to_numpy(), np.full(5, last_close))
    np.testing.assert_allclose(forecast["predicted_return"].to_numpy(), np.zeros(5))


def test_forecast_future_random_forest_runs_and_produces_finite_prices():
    raw = _synthetic_ohlcv()
    forecast = forecast_future(raw, "random_forest", {"n_estimators": 20}, future_days=5)

    assert len(forecast) == 5
    assert np.isfinite(forecast["predicted_close"]).all()
    assert (forecast["predicted_close"] > 0).all()


def test_add_uncertainty_bounds_widen_further_into_the_future():
    forecast = pd.DataFrame(
        {"predicted_close": [100.0, 101.0, 102.0, 103.0]},
        index=pd.bdate_range("2024-01-02", periods=4),
    )

    bounded = add_uncertainty_bounds(forecast, daily_error=0.01)

    widths = bounded["upper_bound"] - bounded["lower_bound"]
    assert widths.is_monotonic_increasing
    np.testing.assert_allclose(
        (bounded["upper_bound"] + bounded["lower_bound"]) / 2, bounded["predicted_close"]
    )
