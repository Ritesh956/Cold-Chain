import numpy as np
import pandas as pd
import pytest
from coldchain_ml.forecast.baselines import HORIZON, moving_average, seasonal_mean, seasonal_naive
from coldchain_ml.forecast.data import Demand
from coldchain_ml.forecast.features import FEATURES, Context, origin_frame, training_frame
from coldchain_ml.forecast.metrics import bias, mase, naive_scale, rmse, wape
from coldchain_ml.slotting.rules import abc_classes, abc_xyz, item_features, xyz_classes


def make_demand(n_items: int = 4, n_days: int = 260, seed: int = 0) -> Demand:
    rng = np.random.default_rng(seed)
    week = np.array([1.0, 0.9, 0.9, 0.9, 1.0, 1.2, 1.3])
    base = rng.uniform(3, 30, size=(n_items, 1))
    units = rng.poisson(base * np.tile(week, n_days // 7 + 1)[:n_days]).astype(np.float32)
    price = np.tile(rng.uniform(1, 5, size=(n_items, 1)), (1, n_days)).astype(np.float32)
    price[0, :20] = np.nan  # item 0 launches on day 20
    units[0, :20] = 0
    dates = pd.date_range("2014-01-06", periods=n_days)  # a Monday
    cal = pd.DataFrame(
        {
            "dow": dates.dayofweek.to_numpy(), "dom": dates.day.to_numpy(), "month": dates.month.to_numpy(),
            "snap_share": np.zeros(n_days), "event": np.zeros(n_days, dtype=np.int8),
            "event_type": np.zeros(n_days, dtype=np.int8),
        }
    )  # fmt: skip
    items = pd.DataFrame(
        {"item_id": [f"I{i}" for i in range(n_items)], "dept_id": ["D1", "D2"] * (n_items // 2)}
    )
    return Demand(items, pd.DatetimeIndex(dates), units, price, cal)


def test_metrics_known_values():
    y = np.array([[10.0, 0.0], [5.0, 5.0]])
    p = np.array([[8.0, 1.0], [5.0, 7.0]])
    assert wape(y, p) == pytest.approx(5 / 20)
    assert rmse(y, p) == pytest.approx(np.sqrt((4 + 1 + 0 + 4) / 4))
    assert bias(y, p) == pytest.approx(21 / 20 - 1)
    assert mase(y, p, np.array([1.0, 2.0])) == pytest.approx(np.mean([1.5 / 1.0, 1.0 / 2.0]))


def test_naive_scale_skips_prelaunch_zeros_and_flat_items():
    n_days = 100
    u = np.zeros((3, n_days))
    u[0, 30:] = (
        np.arange(70) % 7 + 10.0
    )  # weekly pattern from day 30: seasonal-naive error is exactly 0 there
    u[1, 10:] = np.arange(90) + 1.0  # trend: lag-7 difference is 7 everywhere
    u[2, :] = 5.0
    sc = naive_scale(u, o=99, start=np.array([30, 10, 0]))
    assert np.isnan(sc[0]) and np.isnan(sc[2])  # no variation -> MASE undefined for them
    assert sc[1] == pytest.approx(7.0)


def test_baselines_exact_on_a_pure_weekly_pattern():
    pat = np.array([5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0])
    units = np.tile(np.tile(pat, 40), (2, 1))
    o = 100
    expected = np.stack([units[:, 0] * 0 + pat[(o + h) % 7] for h in range(1, HORIZON + 1)], axis=1)
    np.testing.assert_allclose(seasonal_naive(units, o), expected)
    np.testing.assert_allclose(seasonal_mean(units, o), expected)
    np.testing.assert_allclose(moving_average(units, o), np.full((2, HORIZON), pat.mean()))


def test_baselines_use_nothing_after_the_origin():
    d = make_demand()
    o = 150
    for fn in (seasonal_naive, moving_average, seasonal_mean):
        a = fn(d.units, o)
        u2 = d.units.copy()
        u2[:, o + 1 :] += 999
        np.testing.assert_array_equal(a, fn(u2, o))


def test_features_are_causal_and_cover_only_launched_items():
    d = make_demand()
    o = 150
    a = origin_frame(Context(d), o, with_target=False)
    d2 = make_demand()
    d2.units[:, o + 1 :] += 1000  # the future changes; the past must not matter
    b = origin_frame(Context(d2), o, with_target=False)
    pd.testing.assert_frame_equal(a, b)
    assert list(a.columns[: len(FEATURES)]) == FEATURES
    assert len(a) == 4 * HORIZON and set(a["item"]) == {0, 1, 2, 3}

    early = origin_frame(Context(d), 95, with_target=False)  # all items launched by then
    assert len(early) == 4 * HORIZON
    d.price[1, :140] = np.nan  # item 1 not on sale until day 140: it has no forecast at origin 100
    assert 1 not in set(origin_frame(Context(d), 100, with_target=False)["item"])


def test_training_targets_are_all_observed_by_the_last_origin():
    d = make_demand(n_days=300)
    last = 250
    tr = training_frame(Context(d), last, window=120, stride=10)
    assert (tr["origin"] + tr["h"]).max() <= last
    # the target column really is the sale on day origin + h
    row = tr.iloc[123]
    assert row["y"] == d.units[int(row["item"]), int(row["origin"] + row["h"])]


def test_abc_xyz_rules():
    vol = np.array([80.0, 10.0, 5.0, 3.0, 2.0])  # shares 80, 10, 5, 3, 2 %
    abc = abc_classes(vol)
    assert abc.tolist() == [0, 1, 1, 2, 2]  # an item starts class A/B/C while the faster items hold < 80/95 %
    assert xyz_classes(np.array([0.1, 0.5, 0.7, 1.0, 1.4])).tolist() == [0, 1, 1, 1, 2]
    assert abc_xyz(vol, np.array([0.1, 1.5, 0.7, 0.2, 0.2])).tolist() == [0, 5, 4, 6, 6]


def test_item_features_shape_and_zero_share():
    u = np.zeros((2, 28))
    u[0] = 4.0
    u[1, ::2] = 10.0
    f = item_features(u, 0, 28)
    assert f.shape == (2, 4)
    assert f[0, 3] == 0.0 and f[1, 3] == 0.5  # zero-day share
    assert f[0, 1] == 0.0  # constant weekly volume -> CV 0
