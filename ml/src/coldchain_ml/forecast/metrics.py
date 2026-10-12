"""Forecast metrics. No MAPE: it explodes on low volumes (zero-sale days are common here)."""

from __future__ import annotations

import numpy as np


def wape(y: np.ndarray, pred: np.ndarray) -> float:
    """Weighted absolute percentage error: sum|error| / sum(actual). Volume-weighted, defined at zeros."""
    return float(np.abs(y - pred).sum() / y.sum())


def rmse(y: np.ndarray, pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y - pred) ** 2)))


def bias(y: np.ndarray, pred: np.ndarray) -> float:
    """Total forecast over total actual, minus 1 (positive = over-forecasting)."""
    return float(pred.sum() / y.sum() - 1.0)


def naive_scale(units: np.ndarray, o: int, start: np.ndarray, season: int = 7, span: int = 365) -> np.ndarray:
    """Per item: mean |y_t - y_(t-season)| over the last `span` days up to o, only from the item's first
    on-sale day `start` (zeros before launch are not demand). NaN where the item has no variation, so
    MASE is undefined for it and it is left out of that average."""
    scale = np.full(units.shape[0], np.nan)
    for i in range(units.shape[0]):
        lo = max(int(start[i]), o + 1 - span)
        if o - lo < season + 7:
            continue
        seg = units[i, lo : o + 1]
        s = np.abs(seg[season:] - seg[:-season]).mean()
        if s > 0:
            scale[i] = s
    return scale


def mase(y: np.ndarray, pred: np.ndarray, scale: np.ndarray) -> float:
    """Mean over items of (item MAE / item seasonal-naive in-sample MAE). Items with undefined scale
    are skipped."""
    ok = np.isfinite(scale)
    per_item = np.abs(y - pred).mean(axis=1)[ok] / scale[ok]
    return float(per_item.mean())
