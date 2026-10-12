"""Slotting features and the ABC-XYZ rule baseline.

Thresholds are the textbook ones, fixed before looking at any result:
  ABC by cumulative share of volume: A = the items making up the first 80%, B = the next 15%, C = the last 5%.
  XYZ by the coefficient of variation of weekly volume: X < 0.5, Y 0.5 to 1.0, Z > 1.0.
"""

from __future__ import annotations

import numpy as np

FEATURE_NAMES = ["log_adv", "cv_week", "dow_amp", "zero_share"]
CV_X, CV_Y = 0.5, 1.0
ABC_CUTS = (0.80, 0.95)


def item_features(units: np.ndarray, lo: int, hi: int) -> np.ndarray:
    """(n_items, 4) over days lo..hi-1 (hi - lo must be a multiple of 7):
    log1p mean daily units; CV of weekly units; weekday amplitude (max-min weekday mean over overall mean;
    the window starts on the same weekday every time so columns are weekdays); share of zero days."""
    seg = units[:, lo:hi].astype(np.float64)
    n, days = seg.shape
    assert days % 7 == 0
    weekly = seg.reshape(n, days // 7, 7).sum(axis=2)
    mean = seg.mean(axis=1)
    safe = np.maximum(mean, 1e-9)
    dow_mean = seg.reshape(n, days // 7, 7).mean(axis=1)
    return np.column_stack(
        [
            np.log1p(mean),
            weekly.std(axis=1) / np.maximum(weekly.mean(axis=1), 1e-9),
            (dow_mean.max(axis=1) - dow_mean.min(axis=1)) / safe,
            (seg == 0).mean(axis=1),
        ]
    )


def abc_classes(volume: np.ndarray) -> np.ndarray:
    """0 = A, 1 = B, 2 = C by cumulative share of total volume, fastest item first."""
    order = np.argsort(-volume, kind="stable")
    share = np.cumsum(volume[order]) / volume.sum()
    before = np.concatenate(
        [[0.0], share[:-1]]
    )  # share held by faster items: an item starts class A until 80%
    cls_sorted = np.where(before < ABC_CUTS[0], 0, np.where(before < ABC_CUTS[1], 1, 2))
    out = np.empty(len(volume), dtype=int)
    out[order] = cls_sorted
    return out


def xyz_classes(cv: np.ndarray) -> np.ndarray:
    """0 = X (steady), 1 = Y, 2 = Z (erratic)."""
    return np.where(cv < CV_X, 0, np.where(cv <= CV_Y, 1, 2))


def abc_xyz(volume: np.ndarray, cv: np.ndarray) -> np.ndarray:
    """Cell index 0..8 = 3 * ABC + XYZ."""
    return 3 * abc_classes(volume) + xyz_classes(cv)


def cell_name(i: int) -> str:
    return "ABC"[i // 3] + "XYZ"[i % 3]
