"""Load the M5 FOODS sales and reframe them as daily warehouse outflow per item.

M5 (Walmart, Kaggle) is item x store daily unit sales. A distribution centre ships to the stores,
so its outflow for an item is the sum over stores. That is the series forecast here: units per day
per item, 1,437 FOODS items, 1,941 days (2011-01-29 to 2016-05-22). Sales are not demand: a zero can
be a stock-out, and the data says nothing about pallets (no unit-to-pallet conversion is applied).

Files come from `scripts/download_m5.py` and are never committed.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

M5_DIR = Path("data/real/m5")
CATEGORY = "FOODS"
# stores per state in M5 (CA 4, TX 3, WI 3): SNAP days are state-specific, so the warehouse-level
# signal is the share of stores whose state has a SNAP day.
STORES_PER_STATE = {"CA": 4, "TX": 3, "WI": 3}


@dataclass
class Demand:
    items: pd.DataFrame  # item_id, dept_id (one row per series)
    dates: pd.DatetimeIndex  # one per day
    units: np.ndarray  # (n_items, n_days) float32, units shipped per day (sum over stores)
    price: np.ndarray  # (n_items, n_days) float32, mean shelf price over stores; NaN = not on sale
    calendar: pd.DataFrame  # one row per day: dow, dom, month, snap_share, event, event_type

    @property
    def n_days(self) -> int:
        return self.units.shape[1]

    @property
    def n_items(self) -> int:
        return self.units.shape[0]


def _read_sales(path: Path) -> pd.DataFrame:
    """FOODS rows only, read in chunks with small integer dtypes (the full file is 30k x 1.9k)."""
    head = pd.read_csv(path, nrows=0).columns
    day_cols = [c for c in head if c.startswith("d_")]
    dtypes = {c: "int32" for c in day_cols}
    parts = []
    for chunk in pd.read_csv(path, dtype=dtypes, chunksize=4000):
        parts.append(chunk[chunk["cat_id"] == CATEGORY])
    return pd.concat(parts, ignore_index=True)


def load_demand(root: Path = M5_DIR) -> Demand:
    cal = pd.read_csv(root / "calendar.csv", parse_dates=["date"])
    sales = _read_sales(root / "sales_train_evaluation.csv")
    day_cols = [f"d_{i}" for i in range(1, len(cal.index) + 1) if f"d_{i}" in sales.columns]
    n_days = len(day_cols)
    cal = cal.iloc[:n_days].reset_index(drop=True)

    by_item = sales.groupby("item_id", sort=True)[day_cols].sum()
    items = (
        sales.drop_duplicates("item_id")[["item_id", "dept_id"]]
        .set_index("item_id")
        .loc[by_item.index]
        .reset_index()
    )
    units = by_item.to_numpy(dtype=np.float32)

    prices = pd.read_csv(root / "sell_prices.csv", usecols=["item_id", "wm_yr_wk", "sell_price"])
    prices = prices[prices["item_id"].isin(by_item.index)]
    weekly = prices.groupby(["item_id", "wm_yr_wk"])["sell_price"].mean().unstack("wm_yr_wk")
    weekly = weekly.reindex(index=by_item.index)
    week_of_day = cal["wm_yr_wk"].to_numpy()
    price = weekly.reindex(columns=week_of_day).to_numpy(dtype=np.float32)

    snap = np.zeros(n_days)
    for state, n in STORES_PER_STATE.items():
        snap += n * cal[f"snap_{state}"].to_numpy()
    snap /= sum(STORES_PER_STATE.values())
    event = cal["event_name_1"].notna().to_numpy()
    cal_out = pd.DataFrame(
        {
            "dow": cal["date"].dt.dayofweek.to_numpy(),
            "dom": cal["date"].dt.day.to_numpy(),
            "month": cal["date"].dt.month.to_numpy(),
            "snap_share": snap,
            "event": event.astype(np.int8),
            "event_type": pd.Categorical(cal["event_type_1"].fillna("none")).codes.astype(np.int8),
        }
    )
    return Demand(items, pd.DatetimeIndex(cal["date"]), units, price, cal_out)
