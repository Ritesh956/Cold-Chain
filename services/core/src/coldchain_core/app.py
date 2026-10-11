from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any

from coldchain_schemas import Alert, SensorReading
from fastapi import FastAPI, HTTPException, Query, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from coldchain_core.alerts import AlertEngine
from coldchain_core.cache import LatestCache
from coldchain_core.config import Settings
from coldchain_core.events import AlertPublisher, Broadcaster
from coldchain_core.ingest import IngestWorker, Source
from coldchain_core.metrics import Metrics
from coldchain_core.mlclient import MlApi
from coldchain_core.repo import Repo
from coldchain_core.scoring import MODELS, Scorer

log = logging.getLogger("coldchain_core")


@dataclass
class Deps:
    repo: Repo
    cache: LatestCache
    ml: MlApi
    publisher: AlertPublisher
    broadcaster: Broadcaster
    source: Source | None = None  # None: no NATS ingest (tests, API-only mode)
    nats_ok: Callable[[], bool] = lambda: True
    closers: list[Callable[[], Awaitable[None]]] = field(default_factory=list)


async def _supervise(name: str, make: Callable[[], Awaitable[None]], backoff_s: float = 2.0) -> None:
    """Run a background loop forever; log and restart it if it crashes."""
    while True:
        try:
            await make()
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("%s crashed; restarting in %.1fs", name, backoff_s)
            await asyncio.sleep(backoff_s)


async def build_default_deps(settings: Settings, broadcaster: Broadcaster) -> Deps:
    import nats
    from coldchain_schemas.streams import (
        ALERT_MAX_AGE_S,
        ALERT_STREAM,
        ALERT_SUBJECTS,
        SENSOR_MAX_AGE_S,
        SENSOR_STREAM,
        SENSOR_SUBJECTS,
    )

    from coldchain_core.cache import RedisCache
    from coldchain_core.events import BroadcastAlertPublisher, CompositePublisher, NatsAlertPublisher
    from coldchain_core.ingest import NatsSource
    from coldchain_core.mlclient import HttpMlApi
    from coldchain_core.pg import PgRepo

    repo = PgRepo(settings.database_url)
    await repo.init()
    cache = RedisCache(settings.redis_url)
    nc = await nats.connect(settings.nats_url, max_reconnect_attempts=-1)
    js = nc.jetstream()
    await js.add_stream(name=SENSOR_STREAM, subjects=SENSOR_SUBJECTS, max_age=SENSOR_MAX_AGE_S)
    await js.add_stream(name=ALERT_STREAM, subjects=ALERT_SUBJECTS, max_age=ALERT_MAX_AGE_S)
    ml = HttpMlApi(settings.ml_api_url)
    publisher = CompositePublisher(
        BroadcastAlertPublisher(broadcaster),
        NatsAlertPublisher(js),
        on_error=lambda sink, e: log.warning("alert sink %s failed: %r", type(sink).__name__, e),
    )
    return Deps(
        repo=repo,
        cache=cache,
        ml=ml,
        publisher=publisher,
        broadcaster=broadcaster,
        source=NatsSource(js),
        nats_ok=lambda: nc.is_connected,
        closers=[repo.close, cache.close, ml.close, nc.drain],
    )


def _sse(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def create_app(
    deps: Deps | None = None, settings: Settings | None = None, run_background: bool = True
) -> FastAPI:
    settings = settings or Settings.from_env()
    metrics = Metrics()
    broadcaster = deps.broadcaster if deps else Broadcaster()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        d = deps or await build_default_deps(settings, broadcaster)
        app.state.deps = d
        tasks: list[asyncio.Task] = []
        if run_background:
            scorer = Scorer(
                d.repo, d.ml, AlertEngine(settings.alert_close_after_min), d.publisher, metrics, settings
            )
            app.state.scorer = scorer

            async def score_loop() -> None:
                await scorer.start()
                while True:
                    await scorer.tick()
                    await asyncio.sleep(settings.score_tick_s)

            tasks.append(asyncio.create_task(_supervise("scorer", score_loop)))
            if d.source is not None:
                worker = IngestWorker(
                    d.source,
                    d.repo,
                    d.cache,
                    d.broadcaster,
                    metrics,
                    settings.ingest_batch,
                    settings.ingest_wait_s,
                )
                tasks.append(asyncio.create_task(_supervise("ingest", worker.run)))
        try:
            yield
        finally:
            for t in tasks:
                t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            for close in d.closers:
                try:
                    await close()
                except Exception:
                    log.exception("shutdown step failed")

    app = FastAPI(title="ColdChain Core", version="0.1.0", lifespan=lifespan)
    app.state.metrics = metrics

    @app.middleware("http")
    async def observe(request: Request, call_next):
        start = time.perf_counter()
        response = await call_next(request)
        route = getattr(request.scope.get("route"), "path", "unmatched")
        if route != "/v1/stream":  # long-lived; its duration is not a latency
            metrics.requests.labels(request.method, route, str(response.status_code)).inc()
            metrics.request_seconds.labels(route).observe(time.perf_counter() - start)
        return response

    # ---- ops ---------------------------------------------------------------
    @app.get("/healthz", tags=["ops"])
    def healthz() -> dict:
        return {"status": "ok"}

    @app.get("/readyz", tags=["ops"])
    async def readyz(request: Request) -> Response:
        d: Deps = request.app.state.deps
        checks = {"database": await d.repo.ping(), "cache": await d.cache.ping(), "nats": d.nats_ok()}
        ok = all(checks.values())
        return JSONResponse({"ready": ok, "checks": checks}, status_code=200 if ok else 503)

    @app.get("/metrics", tags=["ops"])
    def prometheus(request: Request) -> Response:
        metrics.sse_subscribers.set(request.app.state.deps.broadcaster.subscribers)
        return Response(generate_latest(metrics.registry), media_type=CONTENT_TYPE_LATEST)

    # ---- data --------------------------------------------------------------
    @app.get("/v1/zones", tags=["data"])
    async def zones(request: Request) -> list[dict]:
        d: Deps = request.app.state.deps
        active = await d.repo.open_alerts()
        out = []
        for zone in await d.repo.zones():
            latest = await d.cache.get_latest(zone) or await d.repo.latest(zone)
            out.append(
                {
                    "zone": zone,
                    "latest": latest.model_dump(mode="json") if latest else None,
                    "active_alerts": [
                        {"kind": a.kind.value, "severity": a.severity, "opened_at": a.opened_at}
                        for a in active
                        if a.zone == zone
                    ],
                    "last_scored": {m: await d.repo.get_scoring_state(zone, m) for m in MODELS},
                }
            )
        return json.loads(json.dumps(out, default=lambda o: o.isoformat()))

    @app.get("/v1/zones/{zone}/readings", tags=["data"])
    async def readings(zone: str, request: Request, minutes: int = Query(360, ge=1, le=10080)) -> dict:
        d: Deps = request.app.state.deps
        latest = await d.repo.latest(zone)
        if latest is None:
            raise HTTPException(404, f"unknown zone {zone!r}")
        rows = await d.repo.window(zone, latest.ts, minutes)
        start = latest.ts.timestamp() - (minutes - 1) * 60
        rows = [r for r in rows if r.ts.timestamp() >= start]
        return {
            "zone": zone,
            "end": latest.ts,
            "minutes": minutes,
            "readings": [r.model_dump(mode="json") for r in rows],
        }

    @app.get("/v1/alerts", response_model=list[Alert], tags=["data"])
    async def alerts(
        request: Request,
        status: str = Query("active", pattern="^(active|all)$"),
        zone: str | None = None,
        limit: int = Query(50, ge=1, le=500),
    ) -> list[Alert]:
        return await request.app.state.deps.repo.list_alerts(status == "active", zone, limit)

    @app.get("/v1/stream", tags=["data"])
    async def stream(request: Request) -> StreamingResponse:
        b: Broadcaster = request.app.state.deps.broadcaster

        async def gen() -> AsyncIterator[str]:
            async with b.subscribe() as q:
                yield ": connected\n\n"
                while not await request.is_disconnected():
                    try:
                        ev = await asyncio.wait_for(q.get(), timeout=15)
                    except TimeoutError:
                        yield ": keep-alive\n\n"
                        continue
                    yield _sse(ev["type"], ev)

        return StreamingResponse(gen(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})

    return app


def app_factory() -> FastAPI:
    return create_app()


__all__ = ["Deps", "create_app", "app_factory", "SensorReading"]
