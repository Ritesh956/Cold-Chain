from __future__ import annotations

import asyncio
import json
import socket

import httpx
import pytest
import uvicorn
from coldchain_core.app import Deps, create_app
from coldchain_core.cache import MemoryCache
from coldchain_core.config import Settings
from coldchain_core.events import BroadcastAlertPublisher, Broadcaster
from coldchain_core.repo import MemoryRepo
from coldchain_schemas import Alert, AlertKind
from core_helpers import MIN, T0, FakeMl, history
from fastapi.testclient import TestClient


def make_deps(repo=None, cache=None) -> Deps:
    b = Broadcaster()
    return Deps(
        repo=repo or MemoryRepo(),
        cache=cache or MemoryCache(),
        ml=FakeMl(),
        publisher=BroadcastAlertPublisher(b),
        broadcaster=b,
    )


@pytest.fixture()
def seeded():
    deps = make_deps()
    asyncio.run(deps.repo.upsert_readings([*history("A", 120), *history("B", 30, temp=-20.0)]))
    asyncio.run(
        deps.repo.save_alert(
            Alert(zone="A", kind=AlertKind.ANOMALY, severity="warning", opened_at=T0 + 50 * MIN)
        )
    )
    closed = Alert(
        zone="B",
        kind=AlertKind.BREACHED,
        severity="critical",
        opened_at=T0 + 10 * MIN,
        closed_at=T0 + 20 * MIN,
        peak_score=2.0,
    )
    asyncio.run(deps.repo.save_alert(closed))
    asyncio.run(deps.repo.set_scoring_state("A", "anomaly", T0 + 119 * MIN))
    with TestClient(create_app(deps, Settings(), run_background=False)) as c:
        yield c, deps


def test_health_and_ready(seeded):
    c, _ = seeded
    assert c.get("/healthz").json() == {"status": "ok"}
    r = c.get("/readyz")
    assert r.status_code == 200 and r.json()["checks"] == {"database": True, "cache": True, "nats": True}


def test_not_ready_when_a_dependency_is_down():
    class DownRepo(MemoryRepo):
        async def ping(self):
            return False

    with TestClient(create_app(make_deps(DownRepo()), Settings(), run_background=False)) as c:
        r = c.get("/readyz")
        assert r.status_code == 503 and r.json()["checks"]["database"] is False


def test_zones_overview(seeded):
    c, _ = seeded
    zones = {z["zone"]: z for z in c.get("/v1/zones").json()}
    assert set(zones) == {"A", "B"}
    assert zones["A"]["latest"]["temp_c"] == -25.0
    assert [a["kind"] for a in zones["A"]["active_alerts"]] == ["anomaly"]
    assert zones["B"]["active_alerts"] == []  # B's alert is closed
    assert zones["A"]["last_scored"]["anomaly"] is not None and zones["A"]["last_scored"]["breach"] is None


def test_readings_window_and_validation(seeded):
    c, _ = seeded
    r = c.get("/v1/zones/A/readings", params={"minutes": 10}).json()
    assert len(r["readings"]) == 10 and r["readings"][-1]["ts"].startswith("2025-01-01T01:59")
    assert c.get("/v1/zones/Z/readings").status_code == 404
    assert c.get("/v1/zones/A/readings", params={"minutes": 0}).status_code == 422
    assert c.get("/v1/zones/A/readings", params={"minutes": 99999}).status_code == 422


def test_alert_listing_filters(seeded):
    c, _ = seeded
    assert [a["zone"] for a in c.get("/v1/alerts").json()] == ["A"]  # active by default
    allof = c.get("/v1/alerts", params={"status": "all"}).json()
    assert [a["zone"] for a in allof] == ["A", "B"] and allof[0]["closed_at"] is None
    assert [a["zone"] for a in c.get("/v1/alerts", params={"status": "all", "zone": "B"}).json()] == ["B"]
    assert c.get("/v1/alerts", params={"status": "bogus"}).status_code == 422
    assert c.get("/v1/alerts", params={"limit": 0}).status_code == 422


def test_metrics_endpoint_reports_traffic(seeded):
    c, _ = seeded
    c.get("/v1/zones")
    text = c.get("/metrics").text
    assert 'http_requests_total{method="GET",route="/v1/zones",status="200"} 1.0' in text
    assert "core_sse_subscribers 0.0" in text


# --------------------------------------------------------------- live stream (real server)
def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


async def test_sse_stream_delivers_live_events_over_a_real_connection():
    deps = make_deps()
    port = _free_port()
    config = uvicorn.Config(
        create_app(deps, Settings(), run_background=False), host="127.0.0.1", port=port, log_level="warning"
    )
    server = uvicorn.Server(config)
    task = asyncio.create_task(server.serve())
    try:
        for _ in range(100):
            if server.started:
                break
            await asyncio.sleep(0.05)
        got: list[tuple[str, dict]] = []
        client = httpx.AsyncClient(base_url=f"http://127.0.0.1:{port}", timeout=10)
        async with client as c, c.stream("GET", "/v1/stream") as resp:
            assert resp.headers["content-type"].startswith("text/event-stream")
            lines = resp.aiter_lines()
            assert await asyncio.wait_for(anext(lines), 5) == ": connected"
            deps.broadcaster.publish({"type": "reading", "reading": {"zone": "A", "temp_c": -24.5}})
            deps.broadcaster.publish({"type": "alert", "action": "opened", "alert": {"zone": "A"}})
            event = None
            while len(got) < 2:
                line = await asyncio.wait_for(anext(lines), 5)
                if line.startswith("event: "):
                    event = line[7:]
                elif line.startswith("data: "):
                    got.append((event, json.loads(line[6:])))
        assert [e for e, _ in got] == ["reading", "alert"]
        assert got[0][1]["reading"]["temp_c"] == -24.5 and got[1][1]["action"] == "opened"
        await asyncio.sleep(0.2)
        assert deps.broadcaster.subscribers == 0  # disconnect cleans up the subscription
    finally:
        server.should_exit = True
        await asyncio.wait_for(task, 10)


async def test_slow_subscriber_loses_oldest_events_instead_of_blocking():
    b = Broadcaster(queue_size=3)
    async with b.subscribe() as q:
        for i in range(10):
            b.publish({"type": "reading", "i": i})
        assert [q.get_nowait()["i"] for _ in range(q.qsize())] == [7, 8, 9]
        assert b.dropped == 7
