import pandas as pd
import pytest

from cwfc.data import build_targets, clean_readings, find_missing_timestamps, split_train_holdout


def make_readings():
    return pd.DataFrame(
        {
            "timestamp": [0, 0, 300, 300, 600, 600],
            "vm_id": ["a", "b", "a", "b", "a", "b"],
            "min_cpu": [1.0] * 6,
            "max_cpu": [50.0] * 6,
            "avg_cpu": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
        }
    )


def test_targets_on_clean_data():
    targets = build_targets(make_readings())
    assert list(targets["timestamp"]) == [0, 300, 600]
    assert list(targets["active_vm_count"]) == [2, 2, 2]
    assert list(targets["aggregate_cpu_utilization"]) == [15.0, 35.0, 55.0]


def test_duplicate_vm_is_counted_once():
    readings = make_readings()
    duplicate = readings.iloc[[0]]
    targets = build_targets(pd.concat([readings, duplicate], ignore_index=True))
    assert targets.loc[0, "active_vm_count"] == 2
    assert targets.loc[0, "aggregate_cpu_utilization"] == 15.0


def test_invalid_readings_are_removed():
    readings = make_readings()
    readings.loc[0, "avg_cpu"] = None
    readings.loc[2, "avg_cpu"] = 150.0
    clean, report = clean_readings(readings)
    assert len(clean) == 4
    assert report["rows_missing_or_out_of_range"] == 2
    targets = build_targets(readings)
    assert list(targets["active_vm_count"]) == [1, 1, 2]


def test_targets_are_sorted_by_time():
    shuffled = make_readings().iloc[::-1]
    targets = build_targets(shuffled)
    assert targets["timestamp"].is_monotonic_increasing


def test_missing_timestamp_is_reported():
    readings = make_readings()
    extra = pd.DataFrame(
        {"timestamp": [1200], "vm_id": ["a"], "min_cpu": [1.0], "max_cpu": [50.0], "avg_cpu": [10.0]}
    )
    targets = build_targets(pd.concat([readings, extra], ignore_index=True))
    assert find_missing_timestamps(targets) == [900]
    assert find_missing_timestamps(build_targets(readings)) == []


def test_split_keeps_time_order():
    targets = build_targets(make_readings())
    train, holdout = split_train_holdout(targets, holdout_size=1)
    assert len(train) == 2 and len(holdout) == 1
    assert train["timestamp"].max() < holdout["timestamp"].min()


def test_split_rejects_bad_holdout_size():
    targets = build_targets(make_readings())
    with pytest.raises(ValueError):
        split_train_holdout(targets, holdout_size=3)