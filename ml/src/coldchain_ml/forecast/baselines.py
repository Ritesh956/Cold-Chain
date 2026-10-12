"""Baselines every model must beat. Each takes the history up to and including day `o` and returns an
(n_items, HORIZON) forecast for days o+1 .. o+HORIZON. None looks past `o`."""

from __future__ import annotations

import numpy as np

HORIZON = 28


def _same_weekday(units: np.ndarray, o: int, h: int, k: int) -> np.ndarray:
    """Value k weeks before the last observed occurrence of the weekday of day o+h (h in 1..HORIZON)."""
    lag = 7 * (-(-h // 7) + k - 1)  # smallest multiple of 7 that is >= h, plus k-1 more weeks
    return units[:, o + h - lag]


def seasonal_naive(units: np.ndarray, o: int) -> np.ndarray:
    """Same weekday last week (the most recent observed one)."""
    return np.stack([_same_weekday(units, o, h, 1) for h in range(1, HORIZON + 1)], axis=1)


def moving_average(units: np.ndarray, o: int, window: int = 28) -> np.ndarray:
    """Flat forecast: mean of the last `window` days."""
    m = units[:, o + 1 - window : o + 1].mean(axis=1, keepdims=True)
    return np.repeat(m, HORIZON, axis=1)


def seasonal_mean(units: np.ndarray, o: int, weeks: int = 4) -> np.ndarray:
    """Mean of the same weekday over the last `weeks` weeks: the strong classical baseline."""
    cols = [
        np.mean([_same_weekday(units, o, h, k) for k in range(1, weeks + 1)], axis=0)
        for h in range(1, HORIZON + 1)
    ]
    return np.stack(cols, axis=1)


BASELINES = {
    "seasonal naive (last week)": seasonal_naive,
    "moving average (28 d)": moving_average,
    "seasonal mean (4 weeks)": seasonal_mean,
}
