"""Causal rolling features for freezer anomaly detection.

Every feature at minute t uses only data up to t (no look-ahead), per zone.
Ground-truth events are never an input.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MODEL_FEATURES = [
    "temp",
    "d5",
    "rmean_15",
    "rmax_15",
    "rmin_15",
    "rstd_15",
    "rmean_60",
    "rmax_60",
    "slope_30",
    "duty_60",
    "duty_360",
    "duty_1440",
    "cur_on_360",
    "temp_rmean_360",
]


def _zone_features(df: pd.DataFrame) -> pd.DataFrame:
    temp = df["temp_c"].interpolate(limit=5, limit_direction="forward")
    on = df["compressor_on"].astype(float)
    cur_on = df["compressor_current_a"] * on

    out = pd.DataFrame(index=df.index)
    out["temp"] = temp
    out["d5"] = temp - temp.shift(5)
    for w in (15, 60):
        r = temp.rolling(w, min_periods=w // 2)
        out[f"rmean_{w}"] = r.mean()
        out[f"rmax_{w}"] = r.max()
    out["rmin_15"] = temp.rolling(15, min_periods=7).min()
    out["rstd_15"] = temp.rolling(15, min_periods=7).std()
    out["slope_30"] = (temp - temp.shift(30)) / 30.0
    out["duty_60"] = on.rolling(60, min_periods=30).mean()
    out["duty_360"] = on.rolling(360, min_periods=120).mean()
    out["duty_1440"] = on.rolling(1440, min_periods=360).mean()
    on_minutes = on.rolling(360, min_periods=120).sum().replace(0.0, np.nan)
    out["cur_on_360"] = (cur_on.rolling(360, min_periods=120).sum() / on_minutes).ffill()

    # long-window context for the rolling z-score baseline (excludes the current minute)
    long = temp.rolling(360, min_periods=120)
    out["temp_rmean_360"] = long.mean().shift(1)
    out["temp_rstd_360"] = long.std().shift(1)
    return out


def build_features(readings: pd.DataFrame) -> pd.DataFrame:
    """Feature frame aligned (same index) with `readings`. NaN where a window is not yet full."""
    parts = [_zone_features(g) for _, g in readings.groupby("zone", sort=False)]
    return pd.concat(parts).loc[readings.index]
