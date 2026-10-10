import numpy as np
import pandas as pd
from coldchain_ml.anomaly.features import build_features
from coldchain_ml.breach.data import BREACH_C, BREACH_FEATURES, HORIZON, build_breach_frame
from coldchain_ml.breach.eval import evaluate_zone

N = 1200


def _readings(temp):
    n = len(temp)
    return pd.DataFrame(
        {
            "ts": pd.date_range("2025-01-01", periods=n, freq="min", tz="UTC"),
            "zone": "A",
            "temp_c": temp,
            "compressor_on": np.arange(n) % 2 == 0,
            "compressor_current_a": 12.0,
        }
    )


def _ramp():
    t = np.full(N, -25.0)
    t[600:700] = -10.0  # breach at 600..699
    return t


def test_label_and_at_risk_boundaries():
    r = _readings(_ramp())
    f = build_breach_frame(r, build_features(r))
    # minute 570 sees minute 600 inside (t, t+30]; minute 569 does not
    assert f.loc[570, "y"] == 1.0 and f.loc[569, "y"] == 0.0
    assert HORIZON == 30
    assert f.loc[599, "at_risk"] and not f.loc[600, "at_risk"]
    assert not f.loc[705, "at_risk"]  # within 10 min after the breach ends
    assert f.loc[711, "at_risk"]
    assert np.isnan(f["y"].iloc[-1])  # unknown at end of history


def test_features_do_not_look_ahead():
    r = _readings(_ramp())
    r2 = r.copy()
    r2.loc[900:, "temp_c"] = -5.0
    a = build_breach_frame(r, build_features(r))
    b = build_breach_frame(r2, build_features(r2))
    cols = [c for c in BREACH_FEATURES if c in a]
    pd.testing.assert_frame_equal(a[cols].iloc[:850], b[cols].iloc[:850])


def _events():
    return pd.DataFrame(
        {
            "type": pd.Series(dtype=str),
            "start": pd.Series(dtype="datetime64[ns, UTC]"),
            "end": pd.Series(dtype="datetime64[ns, UTC]"),
            "is_anomaly": pd.Series(dtype=bool),
            "zone": pd.Series(dtype=str),
        }
    )


def test_lead_time_and_true_segment():
    r = _readings(_ramp())
    alarm = np.zeros(N, dtype=bool)
    alarm[580:590] = True  # true warning, 20 min before the breach at 600
    alarm[100:110] = True  # false alarm
    m = evaluate_zone(pd.DatetimeIndex(r["ts"]), r["temp_c"].to_numpy(), alarm, _events())
    assert m.n_episodes == 1
    assert m.leads["other"] == [20.0]
    assert (m.n_segments, m.n_true_segments) == (2, 1)
    assert m.warned_frac(10) == 1.0


def test_missed_episode_and_stale_alarm():
    r = _readings(_ramp())
    alarm = np.zeros(N, dtype=bool)
    alarm[400:410] = True  # ended 190 min before the breach: too early to count
    m = evaluate_zone(pd.DatetimeIndex(r["ts"]), r["temp_c"].to_numpy(), alarm, _events())
    assert m.warned_frac(1) == 0.0 and m.n_true_segments == 0
    assert BREACH_C == -18.0
