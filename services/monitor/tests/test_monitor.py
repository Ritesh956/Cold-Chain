from __future__ import annotations

import json
import math

import numpy as np
from coldchain_monitor import reference
from coldchain_monitor.app import create_app
from coldchain_monitor.config import Settings
from coldchain_monitor.drift import make_profile
from coldchain_monitor.metrics import Metrics
from coldchain_monitor.monitor import DriftMonitor
from coldchain_monitor.scores import ScoreTracker
from coldchain_monitor.sources import SourceError
from fastapi.testclient import TestClient

RNG = np.random.default_rng(1)
BOUNDS = [1.0, 2.0, 4.0, math.inf]


def cumulative(counts: list[float]) -> list[float]:
    return list(np.cumsum(counts))


def hist_text(name: str, counts: list[float]) -> str:
    lines = [f"# TYPE {name} histogram"]
    for b, c in zip(BOUNDS, cumulative(counts), strict=True):
        le = "+Inf" if math.isinf(b) else str(b)
        lines.append(f'{name}_bucket{{le="{le}"}} {c}')
    return "\n".join(lines) + "\n"


class FakeSource:
    def __init__(self) -> None:
        self.zone_names = ["A"]
        self.rows: dict[str, list[dict]] = {}
        self.metrics_text = ""
        self.fail_core = False
        self.fail_ml = False

    async def zones(self):
        if self.fail_core:
            raise SourceError("core down")
        return self.zone_names

    async def readings(self, zone, minutes):
        return self.rows[zone]

    async def ml_metrics(self):
        if self.fail_ml:
            raise SourceError("ml down")
        return self.metrics_text

    async def close(self):
        return None


def readings(n: int, mean: float, cur: float = 12.0, on_share: float = 0.6) -> list[dict]:
    temps = RNG.normal(mean, 1.0, n)
    on = RNG.random(n) < on_share
    return [
        {"temp_c": float(t), "compressor_on": bool(o), "compressor_current_a": float(RNG.normal(cur, 1))}
        for t, o in zip(temps, on, strict=True)
    ]


def profiles():
    ref = readings(20_000, -25.0)
    amps = [r["compressor_current_a"] for r in ref]
    return {
        "temp_c": make_profile("temp_c", [r["temp_c"] for r in ref]),
        "compressor_current_a": make_profile("compressor_current_a", amps),
        "compressor_on": make_profile("compressor_on", [float(r["compressor_on"]) for r in ref], n_bins=2),
    }


SETTINGS = Settings(baseline_count=100, score_window_s=100.0, score_min_count=50, min_samples=200)


def build(src):
    m = Metrics()
    return DriftMonitor(src, profiles(), SETTINGS, m, clock=lambda: 1000.0), m


async def test_stable_inputs_are_ok():
    src = FakeSource()
    src.rows["A"] = readings(1440, -25.0)
    mon, _ = build(src)
    rep = await mon.run_once()
    assert {r["state"] for r in rep["inputs"]} == {"ok"}


async def test_shifted_temperature_is_flagged_and_exported():
    src = FakeSource()
    src.rows["A"] = readings(1440, -22.5)
    mon, m = build(src)
    rep = await mon.run_once()
    by_feature = {r["feature"]: r for r in rep["inputs"]}
    assert by_feature["temp_c"]["state"] == "drift"
    assert by_feature["compressor_current_a"]["state"] == "ok"
    assert m.input_drift.labels("A", "temp_c")._value.get() == 1.0
    assert m.input_psi.labels("A", "temp_c")._value.get() > 0.25


async def test_too_few_readings_gives_no_verdict():
    src = FakeSource()
    src.rows["A"] = readings(50, -10.0)  # wildly off, but only 50 samples
    mon, m = build(src)
    rep = await mon.run_once()
    assert {r["state"] for r in rep["inputs"]} == {"insufficient_data"}
    assert all(r["psi"] is None for r in rep["inputs"])
    assert m.input_drift.labels("A", "temp_c")._value.get() == 0.0


async def test_sensor_dropouts_are_skipped_not_counted():
    src = FakeSource()
    rows = readings(1440, -25.0)
    for r in rows[:100]:
        r["temp_c"] = None
    src.rows["A"] = rows
    mon, _ = build(src)
    rep = await mon.run_once()
    temp = next(r for r in rep["inputs"] if r["feature"] == "temp_c")
    assert temp["n"] == 1340 and temp["state"] == "ok"


async def test_source_failure_is_reported_not_raised():
    src = FakeSource()
    src.fail_core = True
    src.fail_ml = True
    mon, m = build(src)
    rep = await mon.run_once()
    assert len(rep["errors"]) == 2 and rep["inputs"] == [] and rep["predictions"] == []
    assert m.check_errors._value.get() == 2


async def test_one_zone_failing_does_not_hide_the_others():
    src = FakeSource()
    src.zone_names = ["A", "B"]
    src.rows["A"] = readings(1440, -25.0)

    async def flaky(zone, minutes):
        if zone == "B":
            raise SourceError("zone B timeout")
        return src.rows[zone]

    src.readings = flaky
    mon, _ = build(src)
    rep = await mon.run_once()
    assert {r["zone"] for r in rep["inputs"]} == {"A"} and len(rep["errors"]) == 1


def tracker() -> ScoreTracker:
    return ScoreTracker(100, 100.0, 50, 0.10, 0.25)


def test_score_baseline_is_built_first():
    t = tracker()
    assert t.observe(0, BOUNDS, cumulative([20, 20, 5, 5])).state == "building_baseline"


def test_score_drift_detected_after_baseline():
    t = tracker()
    t.observe(0, BOUNDS, cumulative([60, 30, 10, 0]))  # 100 observations: baseline frozen
    # 200 s later, 150 new observations that landed in the top buckets only
    res = t.observe(200, BOUNDS, cumulative([60, 30, 10 + 75, 75]))
    assert res.state == "drift" and res.psi > 0.25 and res.n_recent == 150


def test_score_same_distribution_is_ok():
    t = tracker()
    t.observe(0, BOUNDS, cumulative([60, 30, 10, 0]))
    res = t.observe(200, BOUNDS, cumulative([60 + 90, 30 + 45, 10 + 15, 0]))
    assert res.state == "ok" and res.psi < 0.05


def test_score_recent_window_forgets_old_drift():
    t = tracker()
    t.observe(0, BOUNDS, cumulative([60, 30, 10, 0]))
    t.observe(50, BOUNDS, cumulative([60, 30, 10 + 200, 0]))  # a drifted burst
    # 400 s later the burst has left the 100 s window and the stream looks normal again
    t.observe(300, BOUNDS, cumulative([60 + 10, 30 + 5, 210 + 2, 0]))
    res = t.observe(400, BOUNDS, cumulative([60 + 70, 30 + 35, 210 + 12, 0]))
    assert res.state == "ok"


def test_score_counter_reset_restarts_baseline():
    t = tracker()
    t.observe(0, BOUNDS, cumulative([60, 30, 10, 0]))
    res = t.observe(100, BOUNDS, cumulative([5, 3, 1, 0]))  # exporter restarted
    assert res.state == "building_baseline"


async def test_prediction_drift_through_the_monitor():
    src = FakeSource()
    src.rows["A"] = readings(1440, -25.0)
    mon, m = build(src)
    src.metrics_text = hist_text("anomaly_score", [60, 30, 10, 0])
    assert (await mon.run_once())["predictions"][0]["state"] in {"insufficient_data"}
    mon.clock = lambda: 1200.0
    src.metrics_text = hist_text("anomaly_score", [60, 30, 10 + 100, 100])
    rep = await mon.run_once()
    pred = rep["predictions"][0]
    assert pred["model"] == "anomaly" and pred["state"] == "drift"
    assert m.score_drift.labels("anomaly")._value.get() == 1.0


def test_shipped_reference_is_valid():
    p = reference.load()
    assert set(p) == {"temp_c", "compressor_current_a", "compressor_on"}
    assert all(x.n > 100_000 for x in p.values())
    raw = json.loads(reference.DEFAULT_PATH.read_text())
    assert "SYNTHETIC" in raw["note"]


def test_http_surface():
    src = FakeSource()
    src.rows["A"] = readings(1440, -25.0)
    with TestClient(create_app(src, SETTINGS, run_background=False)) as c:
        assert c.get("/healthz").json() == {"status": "ok"}
        assert c.get("/v1/drift").json()["inputs"] == []  # nothing checked yet
        rep = c.post("/v1/drift/check").json()
        assert len(rep["inputs"]) == 3
        assert c.get("/v1/drift").json()["inputs"] == rep["inputs"]
        text = c.get("/metrics").text
        assert 'drift_input_psi{feature="temp_c",zone="A"}' in text
