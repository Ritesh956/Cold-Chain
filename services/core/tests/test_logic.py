from coldchain_core.alerts import AlertEngine
from coldchain_core.windows import build_window
from coldchain_schemas import AlertKind
from conftest import MIN, T0, history, reading


# ---------------------------------------------------------------- windows
def test_window_full_coverage_is_exact_grid():
    rows = history("A", 100)
    w = build_window(rows, T0 + 99 * MIN, 50, 0.95)
    assert len(w) == 50
    assert w[0]["ts"] == (T0 + 50 * MIN).isoformat() and w[-1]["ts"] == (T0 + 99 * MIN).isoformat()
    assert all(r["temp_c"] == -25.0 for r in w)


def test_gap_becomes_null_temp_and_carries_compressor_state_forward():
    rows = [
        reading("A", i, on=(i < 40), cur=12.0 if i < 40 else 0.3) for i in range(100) if i not in (60, 61)
    ]
    w = build_window(rows, T0 + 99 * MIN, 100, 0.95)
    gap = [r for r in w if r["temp_c"] is None]
    assert len(gap) == 2
    # the minute before the gap had the compressor OFF; the gap must not invent 'on' or zero current
    assert all(g["compressor_on"] is False and g["compressor_current_a"] == 0.3 for g in gap)


def test_too_sparse_or_missing_end_minute_returns_none():
    rows = history("A", 100)[::3]  # ~33% coverage
    assert build_window(rows, rows[-1].ts, 100, 0.95) is None
    full = history("A", 100)
    without_end = [r for r in full if r.ts != T0 + 99 * MIN]
    assert build_window(without_end, T0 + 99 * MIN, 100, 0.5) is None  # the scored minute must exist


def test_window_uses_prior_reading_to_carry_state_into_leading_gap():
    rows = [reading("A", 49, on=False, cur=0.3), *[reading("A", i) for i in range(52, 100)]]
    w = build_window(rows, T0 + 99 * MIN, 50, 0.9)  # minutes 50..99; 50 and 51 missing
    assert w[0]["temp_c"] is None and w[0]["compressor_on"] is False


# ----------------------------------------------------------------- alerts
def feed(engine, flags, kind=AlertKind.ANOMALY, zone="A", start=0):
    events = []
    for i, f in enumerate(flags):
        events += engine.observe(
            zone, kind, T0 + (start + i) * MIN, f, score=float(i), summary="s", model_version="1"
        )
    return events


def test_alert_opens_on_first_alarm_and_closes_after_quiet_period():
    e = AlertEngine(close_after_min=3)
    ev = feed(e, [0, 0, 1, 1, 1, 0, 0, 0, 0])
    assert [x.action for x in ev] == ["opened", "closed"]
    opened, closed = ev[0].alert, ev[1].alert
    assert opened.opened_at == T0 + 2 * MIN
    assert closed.closed_at == T0 + 5 * MIN  # first quiet minute after the last alarm
    assert closed.peak_score == 4.0  # highest score seen while open
    assert not e.active


def test_flapping_within_the_quiet_window_is_one_alert():
    e = AlertEngine(close_after_min=3)
    ev = feed(e, [1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0])
    assert [x.action for x in ev] == ["opened", "closed"]  # not three alerts


def test_kinds_and_zones_are_independent_and_severity_is_set():
    e = AlertEngine(close_after_min=2)
    feed(e, [1, 1], AlertKind.ANOMALY, "A")
    feed(e, [1, 1], AlertKind.BREACHED, "A")
    feed(e, [1, 1], AlertKind.ANOMALY, "B")
    active = {(a.zone, a.kind): a.severity for a in e.active}
    assert active == {
        ("A", AlertKind.ANOMALY): "warning",
        ("A", AlertKind.BREACHED): "critical",
        ("B", AlertKind.ANOMALY): "warning",
    }


def test_quiet_with_no_open_alert_does_nothing():
    assert feed(AlertEngine(), [0, 0, 0, 0]) == []


def test_restore_prevents_a_duplicate_alert_after_restart():
    first = AlertEngine(close_after_min=3)
    opened = feed(first, [1, 1])[0].alert.model_copy(update={"id": 7})
    second = AlertEngine(close_after_min=3)
    second.restore([opened])
    assert feed(second, [1, 1], start=2) == []  # still the same open alert
    closed = feed(second, [0, 0, 0], start=4)
    assert [x.action for x in closed] == ["closed"] and closed[0].alert.id == 7


def test_assign_id_flows_into_the_closed_alert():
    e = AlertEngine(close_after_min=2)
    e.observe("A", AlertKind.ANOMALY, T0, True, 1.0, "s", "1")
    e.assign_id("A", AlertKind.ANOMALY, 42)
    out = feed(e, [0, 0], start=1)
    assert out[0].alert.id == 42
