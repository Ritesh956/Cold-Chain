import numpy as np
import pandas as pd
from coldchain_ml.anomaly.events_eval import apply_min_run, evaluate_zone, segments
from coldchain_ml.common.stats import wilson_interval

N = 1000
TS = pd.date_range("2025-01-01", periods=N, freq="min", tz="UTC")


def _events(rows):
    return pd.DataFrame(
        [
            {"type": t, "start": TS[s], "end": TS[e], "is_anomaly": a}
            for t, s, e, a in rows
        ]
    )


def _alarm(*spans):
    a = np.zeros(N, dtype=bool)
    for s, e in spans:
        a[s:e] = True
    return a


def test_segments_merge_small_gaps():
    a = _alarm((10, 20), (23, 30), (100, 110))
    s, e = segments(a, merge_gap=5)
    assert list(s) == [10, 100] and list(e) == [30, 110]
    s, e = segments(a, merge_gap=2)
    assert list(s) == [10, 23, 100]


def test_min_run_drops_blips():
    a = _alarm((10, 12), (50, 60))
    out = apply_min_run(a, 5)
    assert not out[10:12].any() and out[50:60].all()


def test_perfect_detection_with_delay():
    ev = _events([("power_cut", 200, 300, True)])
    m = evaluate_zone(TS, _alarm((215, 260)), ev)
    assert (m.n_events, m.n_detected) == (1, 1)
    assert m.recall == 1.0 and m.precision == 1.0
    assert m.delays_min["power_cut"] == [15.0]


def test_alarm_shortly_after_event_end_still_counts_within_tolerance():
    ev = _events([("door_left_open", 200, 220, True)])
    assert evaluate_zone(TS, _alarm((235, 240)), ev, tol_min=30).n_detected == 1
    assert evaluate_zone(TS, _alarm((300, 305)), ev, tol_min=30).n_detected == 0


def test_missed_event_and_false_alarm_attributed_to_defrost():
    ev = _events([("power_cut", 200, 300, True), ("defrost", 600, 620, False)])
    m = evaluate_zone(TS, _alarm((610, 625)), ev)
    assert m.n_detected == 0 and m.recall == 0.0
    assert m.n_segments == 1 and m.n_false_segments == 1
    assert m.false_by_cause["defrost"] == 1


def test_unexplained_false_alarm():
    ev = _events([("power_cut", 200, 300, True)])
    m = evaluate_zone(TS, _alarm((800, 810)), ev)
    assert m.false_by_cause["unexplained"] == 1
    assert m.false_alarms_per_zone_day == 1 / (N / 1440)


def test_one_long_alarm_over_two_events_counts_both_detected_but_one_segment():
    ev = _events([("door_left_open", 100, 120, True), ("door_left_open", 130, 150, True)])
    m = evaluate_zone(TS, _alarm((100, 150)), ev)
    assert m.n_detected == 2 and m.n_segments == 1 and m.precision == 1.0


def test_wilson_interval_bounds():
    lo, hi = wilson_interval(40, 44)
    assert 0.75 < lo < 0.91 < hi < 0.99
    assert wilson_interval(0, 0) == (0.0, 1.0)
