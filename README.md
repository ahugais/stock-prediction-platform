# Stock Forecast

This is a tool that looks at a stock's price history and tries to guess where it's headed next. Give it a ticker, and it pulls the data, builds a set of technical indicators from it, trains a model, and shows you a projected price for the days ahead.

I want to be upfront about what this is and isn't. It's not something that beats the market, and it's not trying to be investment advice. Nothing that only looks at past prices reliably predicts future ones, and the app actually tells you when a model isn't performing any better than a coin flip. The point of building this was to put together a real, working ML pipeline end to end: pulling live data, engineering features properly (no peeking at the future), testing a model honestly against its own history, and wrapping the whole thing in something people can actually click around in.

## How it works

1. Pull daily price history for the ticker from Yahoo Finance, and cache it locally so you're not re-downloading every run.
2. Turn that into features: moving averages, RSI, rolling volatility, Bollinger Bands, and a few lagged returns.
3. Train a model to predict tomorrow's *return* rather than tomorrow's price directly. Returns behave more consistently across different stocks, and it's the standard way to frame this kind of problem.
4. Test the model the honest way: walk forward day by day through its own past, always training only on data that came before the day it's predicting. This is what the "walk-forward" evaluation is.
5. Once tested, retrain the model on everything available and step it forward one day at a time to project a full future price path, not just a single number.
6. Plot it all, with a shaded band around the forecast that widens the further out you look. It's just the model's own historical error scaled by the square root of how many days out you are, the same idea behind how volatility gets scaled across time horizons in finance.

You can pick between four models: a naive guess (price doesn't change), a moving average, linear regression, or a random forest. They all share the same fit/predict interface, so swapping one out for another is basically a one-line change.

## Running it

Setup:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

The easiest way to use it is the website:

```bash
stock-forecast web
```

Pick a ticker and a date range in the sidebar, hit Predict, and the charts, accuracy numbers, and a downloadable report all show up on the page.

Or run it straight from the terminal:

```bash
stock-forecast run --ticker AAPL --start 2020-01-01 --end 2025-01-01 --model random_forest
```

(if you'd rather not install the console script, `python -m stock_forecast.cli run ...` works the same way)

Flags:

| Flag | Default | What it does |
|---|---|---|
| `--ticker` | required | Ticker symbol, e.g. `AAPL`, `TSLA`, `NVDA` |
| `--start` / `--end` | required | Date range (`YYYY-MM-DD`) |
| `--model` | `random_forest` | `naive`, `moving_average`, `linear_regression`, `random_forest` |
| `--horizon` | `1` | How many days ahead each historical test predicts |
| `--retrain-every` | `21` | How often (in trading days) the model retrains during testing |
| `--min-train-size` | `252` | Minimum history required before testing starts |
| `--n-estimators` | `200` | Number of trees, if using random forest |
| `--future-days` | `7` | How many days beyond today to forecast |
| `--refresh` | off | Re-download data instead of using the local cache |

## Where things end up

```
data/raw/{TICKER}.csv              cached raw price data
data/processed/{TICKER}.csv        engineered features + target
reports/figures/{run_id}/*.html    the interactive charts
reports/runs/{run_id}/report.json  metrics + config, for anything programmatic
reports/runs/{run_id}/report.md    the same thing, readable
```

## Layout

```
src/stock_forecast/
    cli.py             the command-line entry point (run, web, version)
    webapp.py          the Streamlit site, same pipeline behind a browser UI
    config.py          run settings + where files go
    fetch.py           pulls data from Yahoo Finance, handles caching
    features.py        turns prices into model-ready indicators
    backtest.py         the walk-forward testing logic
    forecast.py          projects prices into the future, adds uncertainty
    visualize.py         builds the charts
    report.py            writes out the JSON/Markdown reports
    models/
        baseline.py       naive + moving-average
        linear.py          linear regression
        random_forest.py   random forest
tests/
    test_fetch.py       caching, without hitting the real network
    test_features.py    indicator math, and checking nothing leaks future data
    test_backtest.py    the walk-forward splitting and metric math
    test_forecast.py    the future-projection logic and uncertainty bounds
```

## Tests

```bash
pytest
```

## What I'd add next

Comparing a few models side by side on the same chart, running predictions across a batch of tickers at once, tuning hyperparameters instead of hardcoding them, maybe an XGBoost or LSTM model, wrapping it as a REST API, packaging it with Docker, and eventually deploying the site somewhere with a real URL.
