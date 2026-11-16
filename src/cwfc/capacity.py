"""Turn forecasts into a capacity plan: forecast + headroom learned from past errors."""
from __future__ import annotations

import numpy as np
import pandas as pd


def headroom_from_errors(actual, forecast, quantile: float) -> float:
    """Headroom = the given quantile of under-forecast errors (actual - forecast).

    With quantile=0.9, planning for forecast + headroom covers the actual value
    about 90% of the time, if future errors look like past errors. Never negative.
    """
    if not 0 < quantile < 1:
        raise ValueError("quantile must be between 0 and 1")
    errors = np.asarray(actual, dtype=float) - np.asarray(forecast, dtype=float)
    return max(float(np.quantile(errors, quantile)), 0.0)


def evaluate_plan(actual, planned, under_cost: float = 5.0, over_cost: float = 1.0) -> dict[str, float]:
    """Score a capacity plan against what really happened.

    coverage: share of steps where the plan was enough.
    mean_shortfall: average missing capacity (CPU points) - risk of slowdowns.
    mean_waste: average unused capacity (CPU points) - money spent for nothing.
    cost: under_cost * shortfall + over_cost * waste. Running short is assumed 5x worse by default.
    """
    actual = np.asarray(actual, dtype=float)
    planned = np.asarray(planned, dtype=float)
    shortfall = np.clip(actual - planned, 0, None)
    waste = np.clip(planned - actual, 0, None)
    return {
        "coverage": float((planned >= actual).mean()),
        "mean_shortfall": float(shortfall.mean()),
        "mean_waste": float(waste.mean()),
        "cost": float(under_cost * shortfall.mean() + over_cost * waste.mean()),
    }


def compare_headrooms(
    validation_actual,
    validation_forecast,
    test_actual,
    test_forecast,
    quantiles=(0.5, 0.8, 0.9, 0.95, 0.99),
    under_cost: float = 5.0,
    over_cost: float = 1.0,
) -> pd.DataFrame:
    """Learn a headroom per quantile on validation errors, then score it on the test window."""
    rows = {}
    for q in quantiles:
        headroom = headroom_from_errors(validation_actual, validation_forecast, q)
        planned = np.asarray(test_forecast, dtype=float) + headroom
        rows[f"q{int(round(q * 100))}"] = {
            "headroom": headroom,
            **evaluate_plan(test_actual, planned, under_cost, over_cost),
        }
    return pd.DataFrame(rows).T