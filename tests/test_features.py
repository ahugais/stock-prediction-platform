import numpy as np
import pandas as pd

from stock_forecast.features import FEATURE_COLUMNS, TARGET_COLUMN, build_features


def _synthetic_ohlcv(n: int = 60, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    returns = rng.normal(loc=0.0005, scale=0.01, size=n)
    close = 100 * np.cumprod(1 + returns)
    dates = pd.date_range("2023-01-02", periods=n, freq="B")
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


def test_build_features_has_no_nans():
    features = build_features(_synthetic_ohlcv(), horizon=1)
    assert not features[FEATURE_COLUMNS + [TARGET_COLUMN]].isna().any().any()
    assert len(features) > 0


def test_target_is_next_day_return_with_no_lookahead():
    raw = _synthetic_ohlcv()
    features = build_features(raw, horizon=1)

    expected_target = (raw["Close"].shift(-1) / raw["Close"] - 1).loc[features.index]
    np.testing.assert_allclose(features[TARGET_COLUMN].to_numpy(), expected_target.to_numpy())

    assert raw.index[-1] not in features.index


def test_daily_return_matches_pct_change():
    raw = _synthetic_ohlcv()
    features = build_features(raw, horizon=1)
    expected = raw["Close"].pct_change().loc[features.index]
    np.testing.assert_allclose(features["daily_return"].to_numpy(), expected.to_numpy())


def test_sma_5_matches_rolling_mean():
    raw = _synthetic_ohlcv()
    features = build_features(raw, horizon=1)
    expected = raw["Close"].rolling(5).mean().loc[features.index]
    np.testing.assert_allclose(features["sma_5"].to_numpy(), expected.to_numpy())


def test_multi_day_horizon_shifts_target_further():
    raw = _synthetic_ohlcv()
    features = build_features(raw, horizon=3)
    expected = (raw["Close"].shift(-3) / raw["Close"] - 1).loc[features.index]
    np.testing.assert_allclose(features[TARGET_COLUMN].to_numpy(), expected.to_numpy())
