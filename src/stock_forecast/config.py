"""Run configuration and filesystem layout for the stock-forecast pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
RUNS_DIR = REPORTS_DIR / "runs"


@dataclass
class RunConfig:
    ticker: str
    start: str
    end: str
    model: str = "random_forest"
    horizon: int = 1
    retrain_every: int = 21
    min_train_size: int = 252
    refresh: bool = False
    model_params: dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().strftime("%Y%m%d_%H%M%S"))

    def __post_init__(self) -> None:
        self.ticker = self.ticker.upper().strip()

    @property
    def run_id(self) -> str:
        return f"{self.ticker}_{self.model}_{self.timestamp}"

    def raw_path(self) -> Path:
        return RAW_DIR / f"{self.ticker}.csv"

    def processed_path(self) -> Path:
        return PROCESSED_DIR / f"{self.ticker}.csv"

    def run_dir(self) -> Path:
        return RUNS_DIR / self.run_id

    def figures_dir(self) -> Path:
        return FIGURES_DIR / self.run_id

    def ensure_dirs(self) -> None:
        for path in (RAW_DIR, PROCESSED_DIR, self.run_dir(), self.figures_dir()):
            path.mkdir(parents=True, exist_ok=True)
