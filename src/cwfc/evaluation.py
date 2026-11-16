"""Rolling-origin backtesting on the training window only."""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from cwfc.metrics import mae, rmse

Forecaster = Callable[[np.ndarray, int], np.ndarray]


def rolling_origins(n: int, horizon: int, min_train: int, step: int) -> list[int]:
    """Return forecast origins: each fold trains on [0, origin) and tests on [origin, origin + horizon)."""
    if min_train <= 0 or horizon <= 0 or step <= 0:
        raise ValueError("min_train, horizon and step must be positive")
    return list(range(min_train, n - horizon + 1, step))


def backtest(
    series: np.ndarray,
    forecaster: Forecaster,
    horizon: int = 24,
    min_train: int = 48,
    step: int = 6,
) -> pd.DataFrame:
    """Run the forecaster at each origin and return one row per forecasted point."""
    series = np.asarray(series, dtype=float)
    rows = []
    for fold, origin in enumerate(rolling_origins(len(series), horizon, min_train, step)):
        history = series[:origin]
        actual = series[origin : origin + horizon]
        forecast = forecaster(history, horizon)
        for step_ahead in range(horizon):
            rows.append(
                {
                    "fold": fold,
                    "origin": origin,
                    "step_ahead": step_ahead + 1,
                    "actual": actual[step_ahead],
                    "forecast": forecast[step_ahead],
                }
            )
    return pd.DataFrame(rows)


def compare_models(
    series: np.ndarray,
    candidates: dict[str, Forecaster],
    baseline: str = "persistence",
    **backtest_kwargs,
) -> pd.DataFrame:
    """Backtest every candidate with the same folds and rank them by MAE."""
    results = {}
    for name, forecaster in candidates.items():
        bt = backtest(series, forecaster, **backtest_kwargs)
        fold_mae = (bt["actual"] - bt["forecast"]).abs().groupby(bt["fold"]).mean()
        results[name] = {
            "mae": mae(bt["actual"], bt["forecast"]),
            "rmse": rmse(bt["actual"], bt["forecast"]),
            "mae_fold_std": float(fold_mae.std()),
            "folds_won_vs_baseline": None,
            "_fold_mae": fold_mae,
        }

    baseline_folds = results[baseline]["_fold_mae"]
    for name, result in results.items():
        result["folds_won_vs_baseline"] = int((result["_fold_mae"] < baseline_folds).sum())
        baseline_mae = results[baseline]["mae"]
        if baseline_mae == 0:
            # Constant target: any model with zero error ties the baseline
            result["mae_vs_baseline"] = 1.0 if result["mae"] == 0 else float("inf")
        else:
            result["mae_vs_baseline"] = result["mae"] / baseline_mae
        del result["_fold_mae"]

    return pd.DataFrame(results).T.sort_values("mae")