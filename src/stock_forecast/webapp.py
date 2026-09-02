"""Streamlit dashboard for the stock-forecast pipeline.

Launch with:
    stock-forecast web

or directly with:
    streamlit run src/stock_forecast/webapp.py
"""

from __future__ import annotations

import datetime as dt

import streamlit as st

from stock_forecast.backtest import walk_forward_validate
from stock_forecast.config import RunConfig
from stock_forecast.features import FEATURE_COLUMNS, TARGET_COLUMN, build_features, save_features
from stock_forecast.fetch import fetch_price_history
from stock_forecast.forecast import add_uncertainty_bounds, forecast_future
from stock_forecast.models import MODEL_REGISTRY, get_model
from stock_forecast.report import build_report
from stock_forecast.visualize import build_figures, save_figures

MODEL_LABELS = {
    "naive": "Simple Guess (price stays the same)",
    "moving_average": "Recent Average",
    "linear_regression": "Linear Regression",
    "random_forest": "Random Forest (recommended)",
}

BACKTEST_HORIZON = 1
RETRAIN_EVERY = 21
MIN_TRAIN_SIZE = 252
RANDOM_FOREST_TREES = 200

st.set_page_config(page_title="Stock Forecast", page_icon="📈", layout="wide")

st.title("📈 Stock Forecast")
st.caption("Pick a stock, and this tool predicts what its price might do next based on its past.")

with st.sidebar:
    st.header("Settings")
    ticker = st.text_input("Stock symbol", value="AAPL", help="e.g. AAPL, TSLA, MSFT, NVDA").upper().strip()

    col1, col2 = st.columns(2)
    start = col1.date_input("History start", value=dt.date.today() - dt.timedelta(days=365 * 3))
    end = col2.date_input("History end", value=dt.date.today())

    model_options = sorted(MODEL_REGISTRY)
    model_name = st.selectbox(
        "Prediction method",
        model_options,
        index=model_options.index("random_forest"),
        format_func=lambda key: MODEL_LABELS[key],
    )
    future_days = st.slider("Days to predict into the future", min_value=1, max_value=30, value=7)

    run_clicked = st.button("Predict", type="primary", width="stretch")

if not run_clicked:
    st.info("Choose a stock in the sidebar, then click **Predict**.")
    st.stop()

if not ticker:
    st.error("Enter a stock symbol.")
    st.stop()

model_params = {"n_estimators": RANDOM_FOREST_TREES} if model_name == "random_forest" else {}
config = RunConfig(
    ticker=ticker,
    start=str(start),
    end=str(end),
    model=model_name,
    horizon=BACKTEST_HORIZON,
    retrain_every=RETRAIN_EVERY,
    min_train_size=MIN_TRAIN_SIZE,
    model_params=model_params,
)
config.ensure_dirs()

try:
    with st.spinner(f"Getting {config.ticker}'s history..."):
        raw = fetch_price_history(config.ticker, config.start, config.end)

    with st.spinner("Looking for patterns..."):
        features = build_features(raw, horizon=config.horizon)
        save_features(features, config.processed_path())

    with st.spinner("Checking how well this would have worked in the past..."):
        predictions, metrics = walk_forward_validate(
            features,
            feature_cols=FEATURE_COLUMNS,
            target_col=TARGET_COLUMN,
            model_factory=lambda: get_model(model_name, **model_params),
            min_train_size=config.min_train_size,
            retrain_every=config.retrain_every,
        )

    with st.spinner(f"Predicting the next {future_days} days..."):
        forecast = forecast_future(raw, model_name, model_params, future_days)
        forecast = add_uncertainty_bounds(forecast, daily_error=metrics["mae"])

    with st.spinner("Drawing the charts..."):
        figures = build_figures(features, predictions, config.ticker, raw=raw, forecast_df=forecast)
        figure_paths = save_figures(figures, config.figures_dir())

    report = build_report(config, metrics, figure_paths, forecast=forecast)
except ValueError as exc:
    st.error(str(exc))
    st.stop()

st.success(f"Done predicting {config.ticker}")

last_price = float(raw["Close"].iloc[-1])
final_price = float(forecast["predicted_close"].iloc[-1])
change_pct = (final_price / last_price - 1) * 100

st.subheader("What might happen next")
c1, c2, c3 = st.columns(3)
c1.metric("Price today", f"${last_price:,.2f}")
c2.metric(f"Predicted price in {future_days} days", f"${final_price:,.2f}", f"{change_pct:+.1f}%")
c3.metric("Got the direction right (in the past)", f"{metrics['directional_accuracy']:.0%}")

if metrics["directional_accuracy"] <= 0.5:
    st.warning(
        f"Heads up: when tested against {config.ticker}'s real past, this model only guessed "
        f"up-or-down correctly {metrics['directional_accuracy']:.0%} of the time — that's no better "
        "than flipping a coin. Treat the forecast below as a rough guess, not something to rely on."
    )

st.plotly_chart(figures["future_forecast"], width="stretch")
st.caption(
    "The solid line is what actually happened. The dashed line is this tool's guesses — both for "
    "recent past days, so you can see how well it tracks reality, and for the future days ahead. "
    "The shaded area is how uncertain each guess is — it gets wider the further out the guess goes, "
    "since tomorrow is easier to guess than three weeks from now. It's a projection, not a promise."
)

with st.expander("See the day-by-day predicted prices"):
    forecast_display = forecast.copy()
    for column in ("predicted_close", "lower_bound", "upper_bound"):
        forecast_display[column] = forecast_display[column].round(2)
    forecast_display["predicted_return"] = (forecast_display["predicted_return"] * 100).round(2)
    forecast_display = forecast_display.rename(
        columns={
            "predicted_close": "Predicted Price ($)",
            "predicted_return": "Predicted Change (%)",
            "lower_bound": "Low End ($)",
            "upper_bound": "High End ($)",
        }
    )
    st.dataframe(forecast_display, width="stretch")

st.divider()
st.subheader("How trustworthy is this?")
st.caption(
    "Before predicting the future, this tool tested itself against the stock's real past — "
    "pretending it didn't know what happened next, guessing anyway, then checking the answer."
)

m1, m2 = st.columns(2)
m1.metric("Guessed the up/down direction correctly", f"{metrics['directional_accuracy']:.0%}")
m2.metric("Typical size of a wrong guess", f"{metrics['mae'] * 100:.2f}%")

tab1, tab2, tab3 = st.tabs(["Price History", "Past Predictions vs Reality", "Accuracy Check"])
with tab1:
    st.plotly_chart(figures["price_history"], width="stretch")
with tab2:
    st.plotly_chart(figures["predicted_vs_actual_price"], width="stretch")
with tab3:
    st.plotly_chart(figures["predicted_vs_actual_return"], width="stretch")

st.divider()
col_a, col_b = st.columns(2)
col_a.download_button(
    "Download full report (JSON)",
    data=(config.run_dir() / "report.json").read_bytes(),
    file_name=f"{config.run_id}_report.json",
    mime="application/json",
    width="stretch",
)
col_b.download_button(
    "Download full report (Markdown)",
    data=(config.run_dir() / "report.md").read_bytes(),
    file_name=f"{config.run_id}_report.md",
    mime="text/markdown",
    width="stretch",
)

with st.expander("Raw prediction data from the past-performance test"):
    st.dataframe(predictions, width="stretch")
