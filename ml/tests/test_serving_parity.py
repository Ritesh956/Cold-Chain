"""Train/serve skew guard: features from a trailing window == features from the full history."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from coldchain_ml.anomaly.features import MODEL_FEATURES, build_features
from coldchain_ml.breach.data import BREACH_FEATURES, build_breach_frame
from coldchain_ml.contract import ANOMALY_MIN_HISTORY, BREACH_MIN_HISTORY
from coldchain_simulator import SimConfig, generate

CFG = replace(
    SimConfig(), days=14, zones=("A",), n_door_left_open=2, n_power_cut=2, n_compressor_drift=1, seed=7
)


@pytest.fixture(scope="module")
def full():
    readings, _ = generate(CFG)
    feats = build_features(readings)
    return readings, feats, build_breach_frame(readings, feats)


def _check(readings, feats, frame, window, columns, use_frame):
    rng = np.random.default_rng(0)
    # include minutes inside drift/power-cut/door periods as well as quiet ones
    ts = rng.integers(window + 5, len(readings), size=60)
    for t in ts:
        w = readings.iloc[t - window + 1 : t + 1].reset_index(drop=True)
        wf = build_features(w)
        got = build_breach_frame(w, wf).iloc[-1] if use_frame else wf.iloc[-1]
        want = (frame if use_frame else feats).iloc[t]
        for c in columns:
            if pd.isna(want[c]):
                assert pd.isna(got[c]), (t, c)
            else:
                assert got[c] == pytest.approx(want[c], rel=1e-9, abs=1e-9), (t, c, got[c], want[c])


def test_anomaly_features_match_with_min_history(full):
    readings, feats, frame = full
    _check(readings, feats, frame, ANOMALY_MIN_HISTORY, ["temp", "cur_on_360"], use_frame=False)


def test_breach_features_match_with_min_history(full):
    readings, feats, frame = full
    _check(readings, feats, frame, BREACH_MIN_HISTORY, [*BREACH_FEATURES, "at_risk"], use_frame=True)


def test_shorter_window_is_NOT_equivalent_for_breach_features(full):
    """Documents why the contract needs 1500 minutes: a 500-minute window changes long-window features."""
    readings, feats, frame = full
    t = 9000
    w = readings.iloc[t - 499 : t + 1].reset_index(drop=True)
    got = build_breach_frame(w, build_features(w)).iloc[-1]
    assert (pd.isna(got["duty_1440"]) and not pd.isna(frame.iloc[t]["duty_1440"])) or got[
        "duty_1440"
    ] != pytest.approx(frame.iloc[t]["duty_1440"])


def test_model_feature_subset():
    assert set(MODEL_FEATURES) <= set(BREACH_FEATURES)


def test_run_length_cap_makes_long_runs_window_independent():
    """A compressor running non-stop for 50 h: without the cap, a 1500-min window would report
    on_run=1500 while training saw 3000. With the cap both saturate at the same value."""
    n = 4000
    readings = pd.DataFrame(
        {
            "ts": pd.date_range("2025-01-01", periods=n, freq="min", tz="UTC"),
            "zone": "A",
            "temp_c": -22.0,
            "compressor_on": np.arange(n) >= 500,  # on from minute 500 to the end (3500 min)
            "compressor_current_a": 15.0,
        }
    )
    full_row = build_breach_frame(readings, build_features(readings)).iloc[-1]
    w = readings.iloc[-BREACH_MIN_HISTORY:].reset_index(drop=True)
    win_row = build_breach_frame(w, build_features(w)).iloc[-1]
    assert full_row["on_run"] == win_row["on_run"] == 1440
