import numpy as np
import pytest

from cwfc.capacity import compare_headrooms, evaluate_plan, headroom_from_errors


def test_headroom_is_quantile_of_under_forecast_errors():
    actual = np.arange(1, 11, dtype=float)
    forecast = np.zeros(10)
    assert headroom_from_errors(actual, forecast, 0.5) == pytest.approx(5.5)


def test_headroom_is_never_negative():
    assert headroom_from_errors([1.0, 1.0], [5.0, 5.0], 0.9) == 0.0


def test_headroom_rejects_bad_quantile():
    with pytest.raises(ValueError):
        headroom_from_errors([1.0], [1.0], 1.0)


def test_evaluate_plan():
    result = evaluate_plan(actual=[10.0, 10.0], planned=[9.0, 12.0], under_cost=5, over_cost=1)
    assert result["coverage"] == 0.5
    assert result["mean_shortfall"] == 0.5
    assert result["mean_waste"] == 1.0
    assert result["cost"] == 3.5


def test_higher_quantile_gives_more_headroom_and_coverage():
    rng = np.random.default_rng(0)
    errors = rng.standard_normal(500)
    table = compare_headrooms(errors, np.zeros(500), errors[:100], np.zeros(100), quantiles=(0.5, 0.9))
    assert table.loc["q90", "headroom"] > table.loc["q50", "headroom"]
    assert table.loc["q90", "coverage"] >= table.loc["q50", "coverage"]
    assert table.loc["q90", "mean_waste"] >= table.loc["q50", "mean_waste"]