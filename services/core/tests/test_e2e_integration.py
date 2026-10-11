"""End to end against REAL NATS JetStream, Postgres and Redis (the CI `integration` job).

Simulator history -> JetStream -> ingest worker -> Postgres/Redis -> scorer (fake ml-api) ->
alerts in Postgres and on the ALERTS stream. Skipped unless NATS_URL, DATABASE_URL and REDIS_URL
are all set. The ml-api is faked here (its models are covered by the ml-api tests); what this
proves is the plumbing: delivery, idempotency, windowing, alert lifecycle and persistence.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
from dataclasses import replace
from datetime import datetime

import pytest
from coldchain_core.alerts import AlertEngine
from coldchain_core.app import build_default_deps, create_app
from coldchain_core.config import Settings
from coldchain_core.events import Broadcaster
from coldchain_schemas import AlertKind, SensorReading, subject_for
from coldchain_schemas.streams import ALERT_STREAM, SENSOR_STREAM
from coldchain_simulator import SimConfig, generate
from core_helpers import MIN, FakeMl

pytestmark = pytest.mark.integration

NATS_URL, DATABASE_URL, REDIS_URL = (os.getenv(k) for k in ("NATS_URL", "DATABASE_URL", "REDIS_URL"))
needs_infra = pytest.mark.skipif(not (NATS_URL and DATABASE_URL and REDIS_URL), reason="infra env not set")

SETTINGS = Settings(
    nats_url=NATS_URL or "",
    database_url=DATABASE_URL or "",
    redis_url=REDIS_URL or "",
    score_tick_s=0.1,
    ingest_wait_s=0.2,
    ingest_batch=500,
    max_chunk_min=120,
    alert_close_after_min=5,
)
HOT = -20.0  # the fake ml-api alarms whenever the temperature is above this


async def wait_for(predicate, timeout=60.0, every=0.2):
    end = asyncio.get_event_loop().time() + timeout
    while asyncio.get_event_loop().time() < end:
        if await predicate():
            return
        await asyncio.sleep(every)
    raise AssertionError("condition not met in time")


def expected_alerts(readings, settings, scored_from):
    """The alert episodes the scorer must produce, computed offline with the same engine."""
    eng, out = AlertEngine(settings.alert_close_after_min), []
    for r in readings:
        if r.ts >= scored_from:
            hot = r.temp_c is not None and r.temp_c > HOT
            for ev in eng.observe("A", AlertKind.ANOMALY, r.ts, hot, 9.0 if hot else 1.0, None, "1"):
                out.append((ev.action, ev.alert.opened_at, ev.alert.closed_at))
    return out


@needs_infra
async def test_history_flows_from_nats_to_postgres_and_raises_alerts():
    import nats
    import psycopg

    # a short simulated history with real injected faults (door left open, power cut)
    cfg = replace(
        SimConfig(), days=5, zones=("A",), seed=11, n_door_left_open=2, n_power_cut=1, n_compressor_drift=0
    )
    readings_df, _ = generate(cfg)
    rows = [
        SensorReading(
            ts=r.ts.to_pydatetime(),
            zone="A",
            temp_c=None if r.temp_c != r.temp_c else float(r.temp_c),
            compressor_on=bool(r.compressor_on),
            compressor_current_a=float(r.compressor_current_a),
        )
        for r in readings_df.itertuples()
    ]
    temps = {r.ts: r.temp_c for r in rows}

    def alarm_when(ts: datetime, kind: str) -> bool:
        t = temps.get(ts)
        return kind == "anomaly" and t is not None and t > HOT

    # clean slate: streams, consumer, tables
    admin = await nats.connect(NATS_URL)
    js = admin.jetstream()
    for stream in (SENSOR_STREAM, ALERT_STREAM):
        with contextlib.suppress(Exception):  # absent on a fresh server
            await js.delete_stream(stream)
    with psycopg.connect(DATABASE_URL, autocommit=True) as c:
        c.execute("DROP TABLE IF EXISTS readings, zones, alerts, scoring_state CASCADE")

    broadcaster = Broadcaster()
    deps = await build_default_deps(SETTINGS, broadcaster)
    deps.ml = FakeMl(anomaly_min=30, breach_min=60, alarm_when=alarm_when)
    app = create_app(deps, SETTINGS, run_background=True)

    async with app.router.lifespan_context(app):
        # Deterministic start: begin scoring at the first reading instead of wherever the scorer
        # happens to wake up ("first sight" only scores the newest minute).
        for model in ("anomaly", "breach"):
            await deps.repo.set_scoring_state("A", model, rows[0].ts)

        # alert subscriber on the ALERTS stream, started before any alert can fire
        alerts_sub = await js.subscribe("alerts.>", stream=ALERT_STREAM, ordered_consumer=True)

        # publish in time order (JetStream acks each publish); concurrent publishes could be
        # stored slightly out of order, which would make the scorer's view of a window nondeterministic
        for r in rows:
            await js.publish(subject_for(r), r.model_dump_json().encode())
        last = rows[-1]

        async def stored() -> bool:
            latest = await deps.repo.latest("A")
            return latest is not None and latest.ts == last.ts

        await wait_for(stored)
        await wait_for(lambda: _scored_to(deps, last.ts))

        # 1. all rows landed exactly once
        with psycopg.connect(DATABASE_URL) as c:
            assert c.execute("SELECT count(*) FROM readings WHERE zone = 'A'").fetchone()[0] == len(rows)

        # 2. redelivery is idempotent: republish the first 400 readings
        await asyncio.gather(*(js.publish(subject_for(r), r.model_dump_json().encode()) for r in rows[:400]))
        await asyncio.sleep(1.5)
        with psycopg.connect(DATABASE_URL) as c:
            assert c.execute("SELECT count(*) FROM readings WHERE zone = 'A'").fetchone()[0] == len(rows)

        # 3. the cache holds the newest reading
        assert (await deps.cache.get_latest("A")).ts == last.ts

        # 4. alerts in Postgres equal the offline expectation from the same alarm series
        want = expected_alerts(rows, SETTINGS, rows[0].ts + MIN)
        opened = [w for w in want if w[0] == "opened"]
        stored_alerts = sorted(await deps.repo.list_alerts(False, "A", 1000), key=lambda a: a.opened_at)
        assert len(opened) > 0, "the simulated history should contain hot episodes"
        assert [a.opened_at for a in stored_alerts] == [o[1] for o in opened]
        want_closed = {w[1]: w[2] for w in want if w[0] == "closed"}
        for a in stored_alerts:
            assert a.kind == AlertKind.ANOMALY and a.closed_at == want_closed.get(a.opened_at)

        # 5. every open/close was published on the ALERTS stream
        n_events = len(want)
        got = []
        for _ in range(n_events):
            msg = await asyncio.wait_for(alerts_sub.next_msg(), 10)
            got.append(json.loads(msg.data))
        assert sorted(g["action"] for g in got) == sorted(w[0] for w in want)
        assert {g["alert"]["zone"] for g in got} == {"A"}

    await admin.drain()


async def _scored_to(deps, ts) -> bool:
    a = await deps.repo.get_scoring_state("A", "anomaly")
    b = await deps.repo.get_scoring_state("A", "breach")
    return a == ts and b == ts
