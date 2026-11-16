import numpy as np
import pytest

from cwfc.evaluation import backtest, compare_models, rolling_origins
from cwfc.models import moving_average_forecast, persistence_forecast, ridge_ar_forecast


def test_baselines():
    history = np.array([1.0, 2.0, 3.0, 4.0])
    assert list(persistence_forecast(history, 3)) == [4.0, 4.0, 4.0]
    assert list(moving_average_forecast(history, 2, window=2)) == [3.5, 3.5]


def test_ridge_ar_on_constant_series():
    forecast = ridge_ar_forecast(np.full(50, 7.0), 5)
    assert np.allclose(forecast, 7.0)


def test_ridge_ar_returns_finite_values():
    rng = np.random.default_rng(0)
    history = 9.4 + 0.2 * rng.standard_normal(100)
    forecast = ridge_ar_forecast(history, 24)
    assert forecast.shape == (24,)
    assert np.isfinite(forecast).all()


def test_ridge_ar_rejects_short_history():
    with pytest.raises(ValueError):
        ridge_ar_forecast(np.ones(5), 3, n_lags=6)


def test_rolling_origins_stay_inside_series():
    origins = rolling_origins(n=110, horizon=24, min_train=48, step=6)
    assert origins[0] == 48
    assert origins[-1] + 24 <= 110


def test_backtest_never_sees_the_future():
    series = np.arange(60, dtype=float)
    seen_lengths = []

    def spy(history, horizon):
        seen_lengths.append(len(history))
        assert history[-1] < series[len(history)]
        return np.zeros(horizon)

    bt = backtest(series, spy, horizon=10, min_train=30, step=10)
    assert seen_lengths == [30, 40, 50]
    assert len(bt) == 30


def test_compare_models_ranks_by_mae():
    series = np.concatenate([np.full(60, 5.0), np.full(40, 6.0)])
    table = compare_models(
        series,
        {"persistence": persistence_forecast, "zero": lambda h, n: np.zeros(n)},
        horizon=10,
        min_train=40,
        step=10,
    )
    assert table.index[0] == "persistence"
    assert table.loc["persistence", "mae_vs_baseline"] == 1.0


def test_compare_models_handles_constant_target():
    table = compare_models(
        np.full(60, 10000.0),
        {"persistence": persistence_forecast, "ridge_ar": ridge_ar_forecast},
        horizon=10,
        min_train=30,
        step=10,
    )
    assert (table["mae"] == 0).all()
    assert (table["mae_vs_baseline"] == 1.0).all()