"""Command-line entry point.

    stock-forecast run --ticker AAPL --start 2020-01-01 --end 2025-01-01 --model random_forest

Pipeline: fetch -> engineer features -> walk-forward backtest -> visualize -> report.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from stock_forecast import __version__
from stock_forecast.backtest import walk_forward_validate
from stock_forecast.config import RunConfig
from stock_forecast.features import FEATURE_COLUMNS, TARGET_COLUMN, build_features, save_features
from stock_forecast.fetch import fetch_price_history
from stock_forecast.forecast import add_uncertainty_bounds, forecast_future
from stock_forecast.models import MODEL_REGISTRY, get_model
from stock_forecast.report import build_report
from stock_forecast.visualize import build_figures, save_figures

app = typer.Typer(help="Modular stock forecasting pipeline: fetch, engineer features, backtest, visualize, report.")
console = Console()


@app.command()
def version() -> None:
    """Print the installed stock-forecast version."""
    console.print(f"stock-forecast {__version__}")


@app.command()
def run(
    ticker: str = typer.Option(..., "--ticker", help="Ticker symbol, e.g. AAPL"),
    start: str = typer.Option(..., "--start", help="Start date YYYY-MM-DD"),
    end: str = typer.Option(..., "--end", help="End date YYYY-MM-DD"),
    model: str = typer.Option("random_forest", "--model", help=f"One of: {', '.join(sorted(MODEL_REGISTRY))}"),
    horizon: int = typer.Option(1, "--horizon", help="Days ahead to predict the return for"),
    retrain_every: int = typer.Option(21, "--retrain-every", help="Retrain cadence (trading days) during walk-forward validation"),
    min_train_size: int = typer.Option(252, "--min-train-size", help="Minimum rows of history before the first walk-forward fold"),
    n_estimators: int = typer.Option(200, "--n-estimators", help="Random Forest: number of trees"),
    future_days: int = typer.Option(7, "--future-days", help="Trading days to forecast beyond the last known price"),
    refresh: bool = typer.Option(False, "--refresh", help="Re-download data even if a local cache exists"),
) -> None:
    """Run the full fetch -> features -> backtest -> visualize -> report pipeline for one ticker."""
    if model not in MODEL_REGISTRY:
        available = ", ".join(sorted(MODEL_REGISTRY))
        raise typer.BadParameter(f"Unknown model '{model}'. Available: {available}")

    model_params = {"n_estimators": n_estimators} if model == "random_forest" else {}
    config = RunConfig(
        ticker=ticker,
        start=start,
        end=end,
        model=model,
        horizon=horizon,
        retrain_every=retrain_every,
        min_train_size=min_train_size,
        refresh=refresh,
        model_params=model_params,
    )
    config.ensure_dirs()

    console.print(f"[bold]Fetching[/bold] {config.ticker} from {start} to {end}...")
    raw = fetch_price_history(config.ticker, start, end, refresh=refresh)

    console.print("[bold]Engineering features[/bold]...")
    features = build_features(raw, horizon=horizon)
    save_features(features, config.processed_path())

    console.print(f"[bold]Backtesting[/bold] with model={model} (walk-forward)...")
    predictions, metrics = walk_forward_validate(
        features,
        feature_cols=FEATURE_COLUMNS,
        target_col=TARGET_COLUMN,
        model_factory=lambda: get_model(model, **model_params),
        min_train_size=min_train_size,
        retrain_every=retrain_every,
    )

    console.print(f"[bold]Forecasting[/bold] the next {future_days} trading days...")
    forecast = forecast_future(raw, model, model_params, future_days)
    forecast = add_uncertainty_bounds(forecast, daily_error=metrics["mae"])

    console.print("[bold]Generating visualizations[/bold]...")
    figures = build_figures(features, predictions, config.ticker, raw=raw, forecast_df=forecast)
    figure_paths = save_figures(figures, config.figures_dir())

    console.print("[bold]Writing report[/bold]...")
    build_report(config, metrics, figure_paths, forecast=forecast)

    table = Table(title=f"{config.ticker} — {model} — Walk-Forward Metrics")
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    table.add_row("MAE (return)", f"{metrics['mae']:.6f}")
    table.add_row("RMSE (return)", f"{metrics['rmse']:.6f}")
    table.add_row("Directional Accuracy", f"{metrics['directional_accuracy']:.2%}")
    table.add_row("# Predictions", str(metrics["n_predictions"]))
    console.print(table)

    if metrics["directional_accuracy"] <= 0.5:
        console.print(
            f"[yellow]Heads up:[/yellow] this model only guessed {config.ticker}'s up-or-down direction "
            f"correctly {metrics['directional_accuracy']:.0%} of the time when tested against its real past "
            "— no better than flipping a coin. Treat the forecast below as a rough guess, not something to rely on."
        )

    last_price = float(raw["Close"].iloc[-1])
    forecast_table = Table(title=f"{config.ticker} — Forecast (last known price: ${last_price:,.2f})")
    forecast_table.add_column("Date")
    forecast_table.add_column("Predicted Price", justify="right")
    forecast_table.add_column("Change from Today", justify="right")
    forecast_table.add_column("Uncertainty Range", justify="right")
    for date, row in forecast.iterrows():
        change_pct = (row["predicted_close"] / last_price - 1) * 100
        forecast_table.add_row(
            date.strftime("%Y-%m-%d"),
            f"${row['predicted_close']:,.2f}",
            f"{change_pct:+.1f}%",
            f"${row['lower_bound']:,.2f} – ${row['upper_bound']:,.2f}",
        )
    console.print(forecast_table)

    console.print(f"[green]Report saved to[/green] {config.run_dir()}")
    console.print(f"[green]Figures saved to[/green] {config.figures_dir()}")


@app.command()
def web(
    port: int = typer.Option(8501, "--port", help="Port to serve the dashboard on"),
) -> None:
    """Launch the interactive Streamlit dashboard in your browser."""
    webapp_path = Path(__file__).parent / "webapp.py"
    console.print(f"[bold]Starting dashboard[/bold] at http://localhost:{port}  (Ctrl+C to stop)")
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", str(webapp_path), "--server.port", str(port)],
        check=False,
    )


if __name__ == "__main__":
    app()
