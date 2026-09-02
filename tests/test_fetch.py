import time

import pandas as pd

from stock_forecast import config as config_module
from stock_forecast import fetch


def _fake_download(*args, **kwargs) -> pd.DataFrame:
    dates = pd.date_range("2023-01-02", periods=5, freq="B")
    return pd.DataFrame(
        {
            "Open": [1.0, 2.0, 3.0, 4.0, 5.0],
            "High": [1.0, 2.0, 3.0, 4.0, 5.0],
            "Low": [1.0, 2.0, 3.0, 4.0, 5.0],
            "Close": [1.0, 2.0, 3.0, 4.0, 5.0],
            "Volume": [100, 100, 100, 100, 100],
        },
        index=dates,
    )


def test_fetch_downloads_once_then_uses_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(config_module, "RAW_DIR", tmp_path)

    calls = {"count": 0}

    def spy_download(*args, **kwargs):
        calls["count"] += 1
        return _fake_download(*args, **kwargs)

    monkeypatch.setattr(fetch.yf, "download", spy_download)

    df1 = fetch.fetch_price_history("FAKE", "2023-01-02", "2023-01-06")
    assert calls["count"] == 1
    assert (tmp_path / "FAKE.csv").exists()

    df2 = fetch.fetch_price_history("FAKE", "2023-01-02", "2023-01-06")
    assert calls["count"] == 1  # served from local cache, no second download
    assert list(df1.index) == list(df2.index)
    assert df1["Close"].tolist() == df2["Close"].tolist()


def test_fetch_refresh_forces_new_download(tmp_path, monkeypatch):
    monkeypatch.setattr(config_module, "RAW_DIR", tmp_path)

    calls = {"count": 0}

    def spy_download(*args, **kwargs):
        calls["count"] += 1
        return _fake_download(*args, **kwargs)

    monkeypatch.setattr(fetch.yf, "download", spy_download)

    fetch.fetch_price_history("FAKE", "2023-01-02", "2023-01-06")
    fetch.fetch_price_history("FAKE", "2023-01-02", "2023-01-06", refresh=True)
    assert calls["count"] == 2


def test_fetch_redownloads_once_the_cache_expires(tmp_path, monkeypatch):
    monkeypatch.setattr(config_module, "RAW_DIR", tmp_path)
    monkeypatch.setattr(fetch, "CACHE_TTL_SECONDS", 0)

    calls = {"count": 0}

    def spy_download(*args, **kwargs):
        calls["count"] += 1
        return _fake_download(*args, **kwargs)

    monkeypatch.setattr(fetch.yf, "download", spy_download)

    fetch.fetch_price_history("FAKE", "2023-01-02", "2023-01-06")
    time.sleep(0.01)
    fetch.fetch_price_history("FAKE", "2023-01-02", "2023-01-06")

    assert calls["count"] == 2  # the zero-second TTL expired instantly, so it re-downloaded
