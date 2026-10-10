"""Labels and features for temperature-breach prediction.

Question asked at every minute t: will the OBSERVED temperature exceed BREACH_C at any minute
in (t, t + HORIZON]?  Only "at risk" rows are scored: temperature currently within limits and
no breach in the last RECENT_MIN minutes (predicting a breach that is already happening is not
a prediction).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from coldchain_ml.anomaly.features import MODEL_FEATURES

BREACH_C = -18.0  # frozen-food limit
HORIZON = 30  # minutes ahead
RECENT_MIN = 10

BREACH_FEATURES = [
    *MODEL_FEATURES,
    "gap_c",
    "slope_5",
    "slope_15",
    "off_run",
    "on_run",
]


def _run_length(b: pd.Series) -> pd.Series:
    """Consecutive-True run length ending at each row (0 where False)."""
    grp = (b != b.shift()).cumsum()
    return (b.groupby(grp).cumcount() + 1).where(b, 0)


def _zone_frame(readings: pd.DataFrame, feats: pd.DataFrame) -> pd.DataFrame:
    temp_obs = readings["temp_c"]
    temp = feats["temp"]
    on = readings["compressor_on"].astype(bool)

    out = feats[MODEL_FEATURES].copy()
    out["gap_c"] = BREACH_C - temp
    out["slope_5"] = (temp - temp.shift(5)) / 5.0
    out["slope_15"] = (temp - temp.shift(15)) / 15.0
    out["off_run"] = _run_length(~on).astype(float)
    out["on_run"] = _run_length(on).astype(float)

    breached = (temp_obs > BREACH_C).fillna(False)
    recent = breached.rolling(RECENT_MIN, min_periods=1).max().astype(bool)
    out["at_risk"] = (temp_obs <= BREACH_C) & ~recent

    # max over (t, t + HORIZON]
    future_max = temp_obs[::-1].rolling(HORIZON, min_periods=1).max()[::-1].shift(-1)
    y = (future_max > BREACH_C).astype(float)
    y[future_max.isna()] = np.nan  # end of history: unknown
    out["y"] = y
    return out


def build_breach_frame(readings: pd.DataFrame, feats: pd.DataFrame) -> pd.DataFrame:
    """Per-row frame aligned with `readings`: BREACH_FEATURES + at_risk + y. Causal features only;
    `y` looks forward and is used only as a label."""
    parts = []
    for _, idx in readings.groupby("zone", sort=False).indices.items():
        parts.append(_zone_frame(readings.iloc[idx], feats.iloc[idx]))
    return pd.concat(parts).loc[readings.index]
