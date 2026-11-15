"""Load the raw files and build the two forecasting targets."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

READINGS_FILE = "vm_cpu_readings.csv.gz"
METADATA_FILE = "vm_metadata.csv.gz"
TARGETS = ["active_vm_count", "aggregate_cpu_utilization"]
HOLDOUT_SIZE = 24


def load_readings(data_dir: str | Path) -> pd.DataFrame:
    """Read the CPU readings file (one row per VM per timestamp)."""
    return pd.read_csv(Path(data_dir) / READINGS_FILE)


def load_metadata(data_dir: str | Path) -> pd.DataFrame:
    """Read the VM metadata file (one row per VM)."""
    return pd.read_csv(Path(data_dir) / METADATA_FILE)


def clean_readings(readings: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Keep only valid observations and report what was removed.

    A valid observation has a timestamp, a vm_id and an avg_cpu between 0 and 100.
    If a VM appears twice at the same timestamp, only the first row is kept.
    """
    required = readings[["timestamp", "vm_id", "avg_cpu"]].notna().all(axis=1)
    in_range = readings["avg_cpu"].between(0, 100)
    valid = readings[required & in_range]

    deduplicated = valid.drop_duplicates(["vm_id", "timestamp"], keep="first")

    report = {
        "rows_in": len(readings),
        "rows_missing_or_out_of_range": len(readings) - len(valid),
        "duplicate_rows": len(valid) - len(deduplicated),
        "rows_out": len(deduplicated),
    }
    return deduplicated, report


def build_targets(readings: pd.DataFrame) -> pd.DataFrame:
    """Aggregate valid readings into one row per timestamp, in time order.

    active_vm_count: number of distinct VMs with a valid reading.
    aggregate_cpu_utilization: mean of avg_cpu across those VMs, in percent.
    """
    clean, _ = clean_readings(readings)
    targets = (
        clean.groupby("timestamp")
        .agg(
            active_vm_count=("vm_id", "nunique"),
            aggregate_cpu_utilization=("avg_cpu", "mean"),
        )
        .sort_index()
        .reset_index()
    )
    return targets


def find_missing_timestamps(targets: pd.DataFrame) -> list[int]:
    """Return timestamps missing from the regular grid (empty list if none).

    The step is the most common gap between consecutive timestamps.
    Missing timestamps are reported, not filled: targets are never imputed.
    """
    timestamps = targets["timestamp"]
    if len(timestamps) < 3:
        return []
    step = int(timestamps.diff().mode().iloc[0])
    expected = range(int(timestamps.iloc[0]), int(timestamps.iloc[-1]) + step, step)
    return sorted(set(expected) - set(timestamps))


def split_train_holdout(
    targets: pd.DataFrame, holdout_size: int = HOLDOUT_SIZE
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split in time order: the last `holdout_size` rows are the holdout."""
    if holdout_size <= 0 or holdout_size >= len(targets):
        raise ValueError("holdout_size must be between 1 and len(targets) - 1")
    train = targets.iloc[:-holdout_size].reset_index(drop=True)
    holdout = targets.iloc[-holdout_size:].reset_index(drop=True)
    return train, holdout