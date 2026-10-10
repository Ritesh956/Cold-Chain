from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from coldchain_simulator import SimConfig, generate, point_labels

ONE_MIN = pd.Timedelta(minutes=1)
SMALL = replace(
    SimConfig(), days=30, zones=("A", "B"), n_door_left_open=2, n_power_cut=2, n_compressor_drift=1
)
CLEAN = replace(SMALL, n_door_left_open=0, n_power_cut=0, n_compressor_drift=0)


@pytest.fixture(scope="module")
def small():
    return generate(SMALL)


@pytest.fixture(scope="module")
def clean():
    return generate(CLEAN)


def test_deterministic_for_same_seed(small):
    r2, e2 = generate(SMALL)
    pd.testing.assert_frame_equal(small[0], r2)
    pd.testing.assert_frame_equal(small[1], e2)


def test_different_seed_differs(small):
    r2, _ = generate(replace(SMALL, seed=SMALL.seed + 1))
    assert not small[0]["temp_c"].equals(r2["temp_c"])


def test_shape_and_columns(small):
    readings, _ = small
    assert len(readings) == SMALL.n_steps * len(SMALL.zones)
    assert list(readings.columns) == ["ts", "zone", "temp_c", "compressor_on", "compressor_current_a"]


def test_anomaly_counts_and_no_overlap(small):
    _, events = small
    an = events[events["is_anomaly"]]
    for zone, grp in an.groupby("zone"):
        assert len(grp) == 2 + 2 + 1, zone
        grp = grp.sort_values("start")
        gaps = grp["start"].iloc[1:].values - grp["end"].iloc[:-1].values
        assert (gaps >= np.timedelta64(SMALL.anomaly_guard_min, "m")).all()


def test_benign_events_are_not_anomalies(small):
    _, events = small
    benign = events[~events["is_anomaly"]]
    assert set(benign["type"]) == {"defrost", "routine_door"}


def test_healthy_zone_holds_setpoint(clean):
    """Without injected faults, a zone cycles around the setpoint and stays clear of the -18 C breach line."""
    readings, _ = clean
    t = readings["temp_c"].dropna()
    assert abs(t.mean() - SMALL.setpoint_c) < 1.0
    assert t.max() < -18.5, "benign events must never cross the -18 C breach line"
    assert t.min() > -28.0


def test_sensor_dropout_rate(clean):
    readings, _ = clean
    rate = readings["temp_c"].isna().mean()
    assert 0.0002 < rate < 0.003


def _zone(readings, events, zone, kind):
    r = readings[readings["zone"] == zone].set_index("ts")
    ev = events[(events["zone"] == zone) & (events["type"] == kind)]
    return r, ev


def test_door_left_open_causes_excursion(small):
    readings, events = small
    r, ev = _zone(readings, events, "A", "door_left_open")
    for s, e in zip(ev["start"], ev["end"], strict=True):
        before = r.loc[s - pd.Timedelta(minutes=60) : s, "temp_c"].mean()
        peak = r.loc[s : e + pd.Timedelta(minutes=10), "temp_c"].max()
        assert peak - before > 3.0


def test_power_cut_warms_the_zone(small):
    readings, events = small
    r, ev = _zone(readings, events, "A", "power_cut")
    for s, e in zip(ev["start"], ev["end"], strict=True):
        assert not r.loc[s : e - ONE_MIN, "compressor_on"].any()
        assert r.loc[e - pd.Timedelta(minutes=5) : e - ONE_MIN, "temp_c"].mean() - r.loc[s, "temp_c"] > 2.5


def test_compressor_drift_raises_current_and_duty(small):
    readings, events = small
    r, ev = _zone(readings, events, "A", "compressor_drift")
    s, e = ev["start"].iloc[0], ev["end"].iloc[0]
    day = pd.Timedelta(days=1)
    early = r.loc[s : s + day]
    late = r.loc[e - day : e]
    on_early = early[early["compressor_on"]]["compressor_current_a"].mean()
    on_late = late[late["compressor_on"]]["compressor_current_a"].mean()
    assert on_late > on_early * 1.1
    assert late["compressor_on"].mean() > early["compressor_on"].mean()


def test_point_labels_cover_only_anomalies(small):
    readings, events = small
    labels = point_labels(readings, events)
    an = events[events["is_anomaly"]]
    expected = sum(int((e - s) / ONE_MIN) for s, e in zip(an["start"], an["end"], strict=True))
    assert labels.sum() == expected
    assert 0 < labels.mean() < 0.2
