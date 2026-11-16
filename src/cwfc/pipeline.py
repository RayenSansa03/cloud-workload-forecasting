"""Command line entry point: build targets, fit on the training window, forecast the holdout.

Usage:
    python -m cwfc.pipeline --data challenge_data --output predictions.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from cwfc.data import HOLDOUT_SIZE, TARGETS, build_targets, load_readings, split_train_holdout
from cwfc.models import persistence_forecast, ridge_ar_forecast

OUTPUT_COLUMNS = ["timestamp"] + TARGETS


def forecast_cpu(history: np.ndarray, horizon: int) -> np.ndarray:
    """Model chosen by rolling-origin validation. Falls back to persistence on very short history."""
    try:
        return ridge_ar_forecast(history, horizon, n_lags=3, alpha=1.0)
    except ValueError:
        return persistence_forecast(history, horizon)


FORECASTERS = {
    "active_vm_count": persistence_forecast,
    "aggregate_cpu_utilization": forecast_cpu,
}


def make_predictions(targets: pd.DataFrame, holdout_size: int = HOLDOUT_SIZE) -> pd.DataFrame:
    """Fit on the training window and forecast every holdout timestamp, in order.

    Only the holdout timestamps are read, never the holdout target values.
    """
    train, holdout = split_train_holdout(targets, holdout_size)
    predictions = pd.DataFrame({"timestamp": holdout["timestamp"].to_numpy()})
    for target, forecaster in FORECASTERS.items():
        predictions[target] = forecaster(train[target].to_numpy(), holdout_size)

    # A VM count is a whole number and cannot be negative
    predictions["active_vm_count"] = predictions["active_vm_count"].round().clip(lower=0).astype(int)
    predictions["aggregate_cpu_utilization"] = predictions["aggregate_cpu_utilization"].clip(0, 100)
    return predictions[OUTPUT_COLUMNS]


def validate_predictions(predictions: pd.DataFrame, expected_timestamps: pd.Series) -> None:
    """Check the submission contract from SUBMISSION.md. Raises ValueError on any problem."""
    if list(predictions.columns) != OUTPUT_COLUMNS:
        raise ValueError(f"columns must be {OUTPUT_COLUMNS}")
    if list(predictions["timestamp"]) != list(expected_timestamps):
        raise ValueError("timestamps must match the holdout timestamps, in order")
    if predictions["timestamp"].duplicated().any():
        raise ValueError("timestamps must be unique")
    values = predictions[TARGETS].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("all predictions must be finite numbers")
    if (predictions["active_vm_count"] < 0).any():
        raise ValueError("active_vm_count must be >= 0")
    if not predictions["aggregate_cpu_utilization"].between(0, 100).all():
        raise ValueError("aggregate_cpu_utilization must be between 0 and 100")


def main(argv: list[str] | None = None) -> pd.DataFrame:
    parser = argparse.ArgumentParser(description="Forecast the holdout window and write predictions.csv")
    parser.add_argument("--data", default="challenge_data", help="folder with the release files")
    parser.add_argument("--output", default="predictions.csv", help="path of the CSV to write")
    parser.add_argument("--holdout-size", type=int, default=HOLDOUT_SIZE)
    args = parser.parse_args(argv)

    targets = build_targets(load_readings(args.data))
    predictions = make_predictions(targets, args.holdout_size)
    validate_predictions(predictions, targets["timestamp"].iloc[-args.holdout_size :])

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output, index=False)
    print(f"Wrote {len(predictions)} predictions to {output}")
    return predictions


if __name__ == "__main__":
    main()