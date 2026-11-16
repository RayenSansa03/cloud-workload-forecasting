"""Forecasting models. Each one takes the past values and returns the next `horizon` values."""
from __future__ import annotations

import numpy as np


def persistence_forecast(history: np.ndarray, horizon: int) -> np.ndarray:
    """Repeat the last observed value."""
    history = np.asarray(history, dtype=float)
    return np.full(horizon, history[-1])


def moving_average_forecast(history: np.ndarray, horizon: int, window: int = 12) -> np.ndarray:
    """Repeat the mean of the last `window` values (12 values = 1 hour)."""
    history = np.asarray(history, dtype=float)
    return np.full(horizon, history[-window:].mean())


def ridge_ar_forecast(
    history: np.ndarray, horizon: int, n_lags: int = 6, alpha: float = 1.0
) -> np.ndarray:
    """Linear autoregressive model with ridge penalty, forecast step by step.

    The series is standardized with the history mean and std, so the model
    pulls forecasts back toward the recent level when the lags carry little signal.
    """
    history = np.asarray(history, dtype=float)
    if len(history) <= n_lags + 1:
        raise ValueError("history is too short for the number of lags")

    mean = history.mean()
    std = history.std()
    if std == 0:
        return np.full(horizon, history[-1])
    z = (history - mean) / std

    # Row t holds the n_lags values before t, most recent first
    features = np.column_stack([z[n_lags - lag : len(z) - lag] for lag in range(1, n_lags + 1)])
    target = z[n_lags:]
    weights = np.linalg.solve(
        features.T @ features + alpha * np.eye(n_lags), features.T @ target
    )

    recent = list(z[-n_lags:][::-1])
    forecast = []
    for _ in range(horizon):
        next_value = float(np.dot(weights, recent))
        forecast.append(next_value)
        recent = [next_value] + recent[:-1]
    return np.array(forecast) * std + mean