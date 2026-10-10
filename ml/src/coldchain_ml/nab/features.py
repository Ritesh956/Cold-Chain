"""Univariate causal features for a single temperature series, defined in minutes so they mean
the same thing on 1-minute simulator data and 5-minute NAB data."""

from __future__ import annotations

import math

import pandas as pd

UNIVARIATE_FEATURES = [
    "temp",
    "d5",
    "rmean_15",
    "rmax_15",
    "rmin_15",
    "rstd_15",
    "rmean_60",
    "rmax_60",
    "slope_30",
    "temp_rmean_360",
]


def build_univariate_features(temp: pd.Series, step_min: int) -> pd.DataFrame:
    def n(minutes: int) -> int:  # minutes -> samples
        return max(1, math.ceil(minutes / step_min))

    temp = temp.reset_index(drop=True)
    out = pd.DataFrame(index=temp.index)
    out["temp"] = temp
    out["d5"] = temp - temp.shift(n(5))
    for w in (15, 60):
        r = temp.rolling(n(w), min_periods=max(1, n(w) // 2))
        out[f"rmean_{w}"] = r.mean()
        out[f"rmax_{w}"] = r.max()
    r15 = temp.rolling(n(15), min_periods=max(1, n(15) // 2))
    out["rmin_15"] = r15.min()
    out["rstd_15"] = r15.std()
    out["slope_30"] = (temp - temp.shift(n(30))) / 30.0
    long = temp.rolling(n(360), min_periods=max(1, n(360) // 3))
    out["temp_rmean_360"] = long.mean().shift(1)
    out["temp_rstd_360"] = long.std().shift(1)
    return out
