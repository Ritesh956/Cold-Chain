from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from coldchain_schemas.auth import ApiKeyMiddleware, describe
from fastapi import FastAPI, Request, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from coldchain_monitor import reference
from coldchain_monitor.config import Settings
from coldchain_monitor.metrics import Metrics
from coldchain_monitor.monitor import DriftMonitor
from coldchain_monitor.sources import DataSource, HttpSource

log = logging.getLogger("coldchain_monitor")


def create_app(
    source: DataSource | None = None, settings: Settings | None = None, run_background: bool = True
) -> FastAPI:
    settings = settings or Settings.from_env()
    metrics = Metrics()
    src = source or HttpSource(
        settings.core_url,
        settings.ml_api_url,
        core_key=settings.core_api_key,
        ml_key=settings.ml_api_key,
    )
    monitor = DriftMonitor(src, reference.load(), settings, metrics)

    async def loop() -> None:
        while True:
            try:
                await monitor.run_once()
            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception("drift check crashed; retrying next interval")
            await asyncio.sleep(settings.interval_s)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        task = asyncio.create_task(loop()) if run_background else None
        try:
            yield
        finally:
            if task:
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
            await src.close()

    app = FastAPI(title="ColdChain Drift Monitor", version="0.1.0", lifespan=lifespan)
    app.state.monitor = monitor
    app.add_middleware(ApiKeyMiddleware, key=settings.api_key)
    log.info("monitor: %s", describe(settings.api_key))

    @app.get("/healthz", tags=["ops"])
    def healthz() -> dict:
        return {"status": "ok"}

    @app.get("/metrics", tags=["ops"])
    def prometheus() -> Response:
        return Response(generate_latest(metrics.registry), media_type=CONTENT_TYPE_LATEST)

    @app.get("/v1/drift", tags=["drift"])
    def drift(request: Request) -> dict:
        return request.app.state.monitor.report

    @app.post("/v1/drift/check", tags=["drift"])
    async def check(request: Request) -> dict:
        return await request.app.state.monitor.run_once()

    return app


def app_factory() -> FastAPI:
    return create_app()
