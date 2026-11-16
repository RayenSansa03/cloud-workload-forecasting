import numpy as np
import pandas as pd
import pytest

from cwfc.data import build_targets
from cwfc.pipeline import OUTPUT_COLUMNS, main, make_predictions, validate_predictions


def make_smoke_readings(n_timestamps=40, n_vms=3, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for t in range(n_timestamps):
        for vm in range(n_vms):
            avg = float(np.clip(10 + 3 * rng.standard_normal(), 0, 100))
            rows.append({"timestamp": t * 300, "vm_id": f"vm{vm}", "min_cpu": avg / 2, "max_cpu": avg * 2, "avg_cpu": avg})
    return pd.DataFrame(rows)


def test_main_writes_valid_csv_on_smoke_data(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    make_smoke_readings().to_csv(data_dir / "vm_cpu_readings.csv.gz", index=False)
    output = tmp_path / "out" / "predictions.csv"

    main(["--data", str(data_dir), "--output", str(output)])

    written = pd.read_csv(output)
    assert list(written.columns) == OUTPUT_COLUMNS
    assert len(written) == 24
    assert list(written["timestamp"]) == [t * 300 for t in range(16, 40)]
    assert (written["active_vm_count"] == 3).all()
    assert written["aggregate_cpu_utilization"].between(0, 100).all()


def test_predictions_do_not_use_holdout_values():
    targets = build_targets(make_smoke_readings())
    changed = targets.copy()
    changed.loc[changed.index[-24:], "aggregate_cpu_utilization"] = 99.0
    changed.loc[changed.index[-24:], "active_vm_count"] = 1

    pd.testing.assert_frame_equal(make_predictions(targets), make_predictions(changed))


def test_short_history_falls_back_to_persistence():
    targets = build_targets(make_smoke_readings(n_timestamps=27))
    predictions = make_predictions(targets)
    last_train_cpu = targets["aggregate_cpu_utilization"].iloc[2]
    assert np.allclose(predictions["aggregate_cpu_utilization"], last_train_cpu)


def test_validate_rejects_wrong_timestamps():
    targets = build_targets(make_smoke_readings())
    predictions = make_predictions(targets)
    with pytest.raises(ValueError):
        validate_predictions(predictions.iloc[:-1], targets["timestamp"].iloc[-24:])
    with pytest.raises(ValueError):
        validate_predictions(predictions, targets["timestamp"].iloc[-25:-1])


def test_validate_rejects_out_of_range_cpu():
    targets = build_targets(make_smoke_readings())
    predictions = make_predictions(targets)
    predictions.loc[0, "aggregate_cpu_utilization"] = 120.0
    with pytest.raises(ValueError):
        validate_predictions(predictions, targets["timestamp"].iloc[-24:])