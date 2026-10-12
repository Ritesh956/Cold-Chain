"""Direct multi-horizon features: one row per (item, forecast origin o, horizon h).

Everything about the past is read at or before day `o`. Only things known in advance about the target
day o+h are used: the calendar (weekday, month, events, SNAP days) and the shelf price (M5 publishes
prices for the forecast weeks). `tests/test_forecast.py` checks that changing any sale after `o`
leaves every feature unchanged.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from coldchain_ml.forecast.baselines import HORIZON
from coldchain_ml.forecast.data import Demand

MIN_ORIGIN = 90  # longest look-back window
FEATURES = [
    "h", "dow", "dom", "month", "snap_share", "event", "event_type",
    "mean7", "mean14", "mean28", "mean56", "mean90", "std28", "zero28", "ratio7_28", "ratio28_90",
    "swd1", "swd2", "swd3", "swd4", "swd_mean",
    "price", "price_vs_origin", "price_vs_year", "age_days", "dept",
]  # fmt: skip


class Context:
    """Per-item running sums computed once, so any origin's windows are O(items)."""

    def __init__(self, d: Demand):
        self.d = d
        y = d.units.astype(np.float64)
        z = np.zeros((y.shape[0], 1))
        self.cs = np.concatenate([z, np.cumsum(y, axis=1)], axis=1)
        self.cs2 = np.concatenate([z, np.cumsum(y**2, axis=1)], axis=1)
        self.cz = np.concatenate([z, np.cumsum(y == 0, axis=1)], axis=1)
        on = ~np.isnan(d.price)
        self.first_on = np.where(on.any(axis=1), on.argmax(axis=1), d.n_days)  # first on-sale day
        self.dept = d.items["dept_id"].astype("category").cat.codes.to_numpy()
        self.year_price = self._rolling_median_price()

    def _rolling_median_price(self) -> np.ndarray:
        """Median shelf price over the 365 days up to each day, computed at the 28-day grid and held:
        cheap, and still only uses the past (a day reads the grid point at or before it)."""
        p = self.d.price
        out = np.full_like(p, np.nan)
        for t in range(0, p.shape[1], 28):
            lo = max(0, t - 364)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)  # items with no price yet: all-NaN slice
                m = np.nanmedian(p[:, lo : t + 1], axis=1) if t >= 1 else p[:, 0]
            out[:, t : t + 28] = m[:, None]
        return out

    def window_mean(self, o: int, w: int) -> np.ndarray:
        return (self.cs[:, o + 1] - self.cs[:, o + 1 - w]) / w


def origin_frame(ctx: Context, o: int, with_target: bool) -> pd.DataFrame:
    """Rows for every item that is on sale at day o (items not yet launched have no future to forecast)."""
    d = ctx.d
    n = d.n_items
    active = np.flatnonzero(ctx.first_on <= o)
    h = np.arange(1, HORIZON + 1)
    t = o + h  # target days

    m = {w: ctx.window_mean(o, w) for w in (7, 14, 28, 56, 90)}
    var28 = (ctx.cs2[:, o + 1] - ctx.cs2[:, o - 27]) / 28 - m[28] ** 2
    std28 = np.sqrt(np.maximum(var28, 0))
    zero28 = (ctx.cz[:, o + 1] - ctx.cz[:, o - 27]) / 28

    base = -(-h // 7)  # whole weeks back to the last observed same weekday
    swd = [d.units[:, t - 7 * (base + k - 1)] for k in range(1, 5)]  # each (n, HORIZON)
    swd_mean = np.mean(swd, axis=0)

    price_t = d.price[:, t]
    price_o = d.price[:, [o]]
    yr = ctx.year_price[:, [o]]
    cal = d.calendar.iloc[t]

    def per_item(a: np.ndarray) -> np.ndarray:
        return np.repeat(a[:, None], HORIZON, axis=1)

    with np.errstate(divide="ignore", invalid="ignore"):
        cols = {
            "h": np.tile(h, (n, 1)),
            "dow": np.tile(cal["dow"].to_numpy(), (n, 1)),
            "dom": np.tile(cal["dom"].to_numpy(), (n, 1)),
            "month": np.tile(cal["month"].to_numpy(), (n, 1)),
            "snap_share": np.tile(cal["snap_share"].to_numpy(), (n, 1)),
            "event": np.tile(cal["event"].to_numpy(), (n, 1)),
            "event_type": np.tile(cal["event_type"].to_numpy(), (n, 1)),
            "mean7": per_item(m[7]), "mean14": per_item(m[14]), "mean28": per_item(m[28]),
            "mean56": per_item(m[56]), "mean90": per_item(m[90]),
            "std28": per_item(std28), "zero28": per_item(zero28),
            "ratio7_28": per_item(m[7] / m[28]), "ratio28_90": per_item(m[28] / m[90]),
            "swd1": swd[0], "swd2": swd[1], "swd3": swd[2], "swd4": swd[3], "swd_mean": swd_mean,
            "price": price_t, "price_vs_origin": price_t / price_o, "price_vs_year": price_t / yr,
            "age_days": per_item(np.minimum(o - ctx.first_on, 2000).astype(float)),
            "dept": per_item(ctx.dept.astype(float)),
        }  # fmt: skip
    frame = pd.DataFrame({k: v[active].ravel() for k, v in cols.items()}).astype("float32")
    frame = frame.replace(
        [np.inf, -np.inf], np.nan
    )  # ratios with a zero denominator are missing, not infinite
    frame["item"] = np.repeat(active, HORIZON)
    frame["origin"] = o
    if with_target:
        frame["y"] = d.units[:, t][active].ravel()
    return frame[FEATURES + ["item", "origin"] + (["y"] if with_target else [])]


def training_frame(ctx: Context, last_origin: int, window: int = 730, stride: int = 10) -> pd.DataFrame:
    """Origins s with s + HORIZON <= last_origin, so every target is already observed at last_origin."""
    newest = last_origin - HORIZON
    origins = range(newest, max(MIN_ORIGIN, newest - window) - 1, -stride)
    return pd.concat([origin_frame(ctx, s, True) for s in origins], ignore_index=True)
