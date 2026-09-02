"""Interactive Plotly reporting charts."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go


def price_history_figure(features_df: pd.DataFrame, ticker: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=features_df.index, y=features_df["Close"], name="Price", line=dict(color="#1f77b4")))
    fig.add_trace(go.Scatter(x=features_df.index, y=features_df["sma_20"], name="20-Day Average", line=dict(color="#ff7f0e", dash="dot")))
    fig.add_trace(go.Scatter(x=features_df.index, y=features_df["bb_upper_20"], name="Typical Range (High)", line=dict(color="#888", width=1), showlegend=True))
    fig.add_trace(go.Scatter(x=features_df.index, y=features_df["bb_lower_20"], name="Typical Range (Low)", line=dict(color="#888", width=1), fill="tonexty", fillcolor="rgba(136,136,136,0.1)"))
    fig.update_layout(title=f"{ticker} Price History", xaxis_title="Date", yaxis_title="Price ($)", template="plotly_white")
    return fig


def future_forecast_figure(
    raw: pd.DataFrame,
    predictions: pd.DataFrame,
    forecast_df: pd.DataFrame,
    ticker: str,
    history_days: int = 90,
) -> go.Figure:
    """Predicted line spanning the backtest period through the forecast."""
    actual_history = raw["Close"].tail(history_days)
    last_date, last_price = actual_history.index[-1], float(actual_history.iloc[-1])

    fig = go.Figure()

    if "lower_bound" in forecast_df.columns and "upper_bound" in forecast_df.columns:
        band_x = list(forecast_df.index) + list(forecast_df.index[::-1])
        band_y = list(forecast_df["upper_bound"]) + list(forecast_df["lower_bound"][::-1])
        fig.add_trace(
            go.Scatter(
                x=band_x,
                y=band_y,
                fill="toself",
                fillcolor="rgba(214,39,40,0.15)",
                line=dict(color="rgba(0,0,0,0)"),
                name="Uncertainty Range",
                hoverinfo="skip",
            )
        )

    fig.add_trace(go.Scatter(x=actual_history.index, y=actual_history.values, name="Actual Price", line=dict(color="#1f77b4")))

    past_predicted = predictions["predicted_close"][predictions.index >= actual_history.index[0]]
    predicted_x = list(past_predicted.index) + [last_date] + list(forecast_df.index)
    predicted_y = list(past_predicted.to_numpy()) + [last_price] + list(forecast_df["predicted_close"].to_numpy())
    fig.add_trace(go.Scatter(x=predicted_x, y=predicted_y, name="Predicted Price", line=dict(color="#d62728", dash="dash")))

    fig.update_layout(title=f"{ticker} — Actual vs Predicted, Past and Future", xaxis_title="Date", yaxis_title="Price ($)", template="plotly_white")
    return fig


def prediction_vs_actual_price_figure(predictions: pd.DataFrame, ticker: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=predictions.index, y=predictions["actual_close"], name="What Actually Happened", line=dict(color="#1f77b4")))
    fig.add_trace(go.Scatter(x=predictions.index, y=predictions["predicted_close"], name="What the Model Predicted", line=dict(color="#d62728", dash="dash")))
    fig.update_layout(title=f"{ticker} — Past Predictions vs What Actually Happened", xaxis_title="Date", yaxis_title="Price ($)", template="plotly_white")
    return fig


def prediction_vs_actual_return_figure(predictions: pd.DataFrame, ticker: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=predictions["actual_return"] * 100,
            y=predictions["predicted_return"] * 100,
            mode="markers",
            marker=dict(color="#2ca02c", size=6, opacity=0.6),
            name="Each Day Tested",
        )
    )
    lo = min(predictions["actual_return"].min(), predictions["predicted_return"].min()) * 100
    hi = max(predictions["actual_return"].max(), predictions["predicted_return"].max()) * 100
    fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", line=dict(color="#888", dash="dot"), name="Perfect Guess"))
    fig.update_layout(
        title=f"{ticker} — How Close Were the Guesses?",
        xaxis_title="What Actually Happened (% change)",
        yaxis_title="What the Model Guessed (% change)",
        template="plotly_white",
    )
    return fig


def build_figures(
    features_df: pd.DataFrame,
    predictions: pd.DataFrame,
    ticker: str,
    raw: pd.DataFrame | None = None,
    forecast_df: pd.DataFrame | None = None,
) -> dict[str, go.Figure]:
    figures = {
        "price_history": price_history_figure(features_df, ticker),
        "predicted_vs_actual_price": prediction_vs_actual_price_figure(predictions, ticker),
        "predicted_vs_actual_return": prediction_vs_actual_return_figure(predictions, ticker),
    }
    if raw is not None and forecast_df is not None:
        figures["future_forecast"] = future_forecast_figure(raw, predictions, forecast_df, ticker)
    return figures


def save_figures(figures: dict[str, go.Figure], out_dir: Path) -> dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}
    for name, fig in figures.items():
        path = out_dir / f"{name}.html"
        fig.write_html(path, include_plotlyjs="cdn")
        paths[name] = str(path)
    return paths
