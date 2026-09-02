"""Assembles and persists the per-run JSON + Markdown report."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

import pandas as pd

from stock_forecast.config import RunConfig


def build_report(
    config: RunConfig,
    metrics: dict[str, float],
    figure_paths: dict[str, str],
    forecast: pd.DataFrame | None = None,
) -> dict[str, Any]:
    report = {
        "run_id": config.run_id,
        "ticker": config.ticker,
        "start": config.start,
        "end": config.end,
        "model": config.model,
        "model_params": config.model_params,
        "horizon_days": config.horizon,
        "retrain_every": config.retrain_every,
        "min_train_size": config.min_train_size,
        "metrics": metrics,
        "figures": figure_paths,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }
    if forecast is not None:
        has_bounds = "lower_bound" in forecast.columns and "upper_bound" in forecast.columns
        report["forecast"] = [
            {
                "date": date.strftime("%Y-%m-%d"),
                "predicted_close": row["predicted_close"],
                "predicted_return": row["predicted_return"],
                **(
                    {"lower_bound": row["lower_bound"], "upper_bound": row["upper_bound"]}
                    if has_bounds
                    else {}
                ),
            }
            for date, row in forecast.iterrows()
        ]

    out_dir = config.run_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.json").write_text(json.dumps(report, indent=2))
    (out_dir / "report.md").write_text(_render_markdown(report))
    return report


def _render_markdown(report: dict[str, Any]) -> str:
    metrics = report["metrics"]
    lines = [
        f"# Stock Forecast Report — {report['ticker']}",
        "",
        f"- **Run ID**: {report['run_id']}",
        f"- **Date range**: {report['start']} to {report['end']}",
        f"- **Model**: {report['model']}",
        f"- **Horizon**: {report['horizon_days']} day(s)",
        f"- **Retrain every**: {report['retrain_every']} trading days",
        f"- **Min train size**: {report['min_train_size']}",
        f"- **Generated at**: {report['generated_at']}",
        "",
    ]

    if report.get("forecast"):
        has_bounds = "lower_bound" in report["forecast"][0]
        header = "| Date | Predicted Price | Predicted Change |" + (" Uncertainty Range |" if has_bounds else "")
        divider = "|---|---|---|" + ("---|" if has_bounds else "")
        lines += ["## Forecast", "", header, divider]
        for row in report["forecast"]:
            line = f"| {row['date']} | ${row['predicted_close']:,.2f} | {row['predicted_return']:+.2%} |"
            if has_bounds:
                line += f" ${row['lower_bound']:,.2f} – ${row['upper_bound']:,.2f} |"
            lines.append(line)
        lines.append("")

    lines += [
        "## Metrics",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| MAE (return) | {metrics['mae']:.6f} |",
        f"| RMSE (return) | {metrics['rmse']:.6f} |",
        f"| Directional Accuracy | {metrics['directional_accuracy']:.2%} |",
        f"| # Predictions | {metrics['n_predictions']} |",
        "",
        "## Figures",
        "",
    ]
    for name, path in report["figures"].items():
        lines.append(f"- [{name}]({path})")
    return "\n".join(lines) + "\n"
