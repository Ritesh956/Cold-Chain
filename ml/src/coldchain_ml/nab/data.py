"""Loader for the NAB machine-temperature series (downloaded by scripts/download_nab.py).

The files are untrusted downloads: they are only parsed as CSV/JSON data here, never executed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

NAB_DIR = Path("data/real/nab")
KEY = "realKnownCause/machine_temperature_system_failure.csv"
STEP_MIN = 5
PROBATION_FRAC = 0.15  # NAB convention: first 15% is for fitting; anomalies there are not scored


@dataclass
class NabSeries:
    df: pd.DataFrame  # columns: timestamp, value; regular 5-min grid, sorted, de-duplicated
    windows: list[tuple[pd.Timestamp, pd.Timestamp]]
    labels: list[pd.Timestamp]
    probation_end: int  # first scored row index
    notes: dict[str, int] = field(default_factory=dict)


def load_nab(directory: Path = NAB_DIR) -> NabSeries:
    raw = pd.read_csv(directory / "machine_temperature_system_failure.csv", parse_dates=["timestamp"])
    out_of_order = int((raw["timestamp"].diff() < pd.Timedelta(0)).sum())
    df = raw.sort_values("timestamp", kind="stable")
    n_dup = int(df["timestamp"].duplicated().sum())
    df = df.drop_duplicates("timestamp", keep="first").reset_index(drop=True)

    steps = df["timestamp"].diff().dropna()
    off_grid = int((steps != pd.Timedelta(minutes=STEP_MIN)).sum())

    windows_json = json.loads((directory / "combined_windows.json").read_text(encoding="utf-8"))[KEY]
    labels_json = json.loads((directory / "combined_labels.json").read_text(encoding="utf-8"))[KEY]
    windows = [(pd.Timestamp(a), pd.Timestamp(b)) for a, b in windows_json]
    labels = [pd.Timestamp(t) for t in labels_json]
    return NabSeries(
        df=df,
        windows=windows,
        labels=labels,
        probation_end=int(PROBATION_FRAC * len(df)),
        notes={
            "rows_raw": len(raw),
            "rows_clean": len(df),
            "out_of_order_steps": out_of_order,
            "duplicate_timestamps_dropped": n_dup,
            "steps_not_5min_after_cleaning": off_grid,
        },
    )
