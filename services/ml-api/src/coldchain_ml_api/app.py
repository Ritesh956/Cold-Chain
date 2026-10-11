from __future__ import annotations

import logging
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from coldchain_ml_api.config import Settings
from coldchain_ml_api.errors import InputError
from coldchain_ml_api.metrics import Metrics
from coldchain_ml_api.models import SIGNAL_LABELS
from coldchain_ml_api.schemas import (
    AnomalyPoint,
    AnomalyResponse,
    BreachPoint,
    BreachResponse,
    ModelRef,
    WindowRequest,
)
from coldchain_ml_api.store import Models
from coldchain_ml_api.windows import to_frame

log = logging.getLogger("coldchain_ml_api")


def _f(x: float | None) -> float | None:
    return None if x is None or not np.isfinite(x) else float(x)


def create_app(models: Models | None = None, settings: Settings | None = None) -> FastAPI:
    metrics = Metrics()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        # Fail fast: a service that cannot load its models must not report ready.
        app.state.models = models or Models.load(settings or Settings.from_env())
        for m in (app.state.models.anomaly, app.state.models.breach):
            metrics.model_info.labels(model=m.name, version=m.version).set(1)
            log.info("loaded %s v%s", m.name, m.version)
        yield

    app = FastAPI(title="ColdChain ML API", version="0.1.0", lifespan=lifespan)
    app.state.metrics = metrics

    @app.middleware("http")
    async def observe(request: Request, call_next):
        start = time.perf_counter()
        response = await call_next(request)
        route = getattr(request.scope.get("route"), "path", "unmatched")
        metrics.requests.labels(request.method, route, str(response.status_code)).inc()
        metrics.request_seconds.labels(route).observe(time.perf_counter() - start)
        return response

    @app.exception_handler(InputError)
    async def input_error(_: Request, exc: InputError) -> JSONResponse:
        metrics.rejected.labels(exc.code).inc()
        return JSONResponse(status_code=422, content={"error": {"code": exc.code, "message": exc.message}})

    @app.get("/healthz", tags=["ops"])
    def healthz() -> dict:
        return {"status": "ok"}

    @app.get("/readyz", tags=["ops"])
    def readyz(request: Request) -> Response:
        ready = getattr(request.app.state, "models", None) is not None
        return JSONResponse({"ready": ready}, status_code=200 if ready else 503)

    @app.get("/metrics", tags=["ops"])
    def prometheus() -> Response:
        return Response(generate_latest(metrics.registry), media_type=CONTENT_TYPE_LATEST)

    @app.get("/v1/models", tags=["models"])
    def model_info(request: Request) -> dict:
        m: Models = request.app.state.models
        return {
            x.name: {
                "version": x.version,
                "min_history_min": x.min_history,
                "threshold": x.threshold,
                "data": x.config.get("data"),
                "reported_test": x.config.get("reported_test"),
            }
            for x in (m.anomaly, m.breach)
        }

    @app.post("/v1/anomaly/score", response_model=AnomalyResponse, tags=["models"])
    def anomaly_score(req: WindowRequest, request: Request) -> AnomalyResponse:
        model = request.app.state.models.anomaly
        df = to_frame(req, model.min_history)
        t0 = time.perf_counter()
        res = model.score(df)
        metrics.inference_seconds.labels(model.name).observe(time.perf_counter() - t0)

        points = []
        for i in range(len(df) - req.return_last, len(df)):
            score = _f(res.score[i])
            drivers = {SIGNAL_LABELS.get(s, s): _f(v[i]) for s, v in res.drivers.items()}
            alarm = bool(res.alarm[i])
            summary = None
            known = {k: v for k, v in drivers.items() if v is not None}
            if alarm and known:
                top = max(known, key=lambda k: abs(known[k]))
                summary = (
                    f"Zone {req.zone}: readings are unusual. Largest deviation is {top} "
                    f"({abs(known[top]):.1f} robust standard deviations from normal)."
                )
            points.append(
                AnomalyPoint(
                    ts=res.ts[i].to_pydatetime(), score=score, alarm=alarm, drivers=drivers, summary=summary
                )
            )
            outcome = "alarm" if alarm else ("no_alarm" if score is not None else "insufficient")
            metrics.predictions.labels(model.name, outcome).inc()
        if points[-1].score is not None:
            metrics.anomaly_score.observe(points[-1].score)
        return AnomalyResponse(
            zone=req.zone,
            model=ModelRef(name=model.name, version=model.version),
            threshold=model.threshold,
            min_run_minutes=model.min_run,
            points=points,
        )

    @app.post("/v1/breach/predict", response_model=BreachResponse, tags=["models"])
    def breach_predict(req: WindowRequest, request: Request) -> BreachResponse:
        model = request.app.state.models.breach
        df = to_frame(req, model.min_history)
        t0 = time.perf_counter()
        res = model.predict(df)
        metrics.inference_seconds.labels(model.name).observe(time.perf_counter() - t0)

        points = []
        for i in range(len(df) - req.return_last, len(df)):
            at_risk, alarm = bool(res.at_risk[i]), bool(res.alarm[i])
            risk = _f(res.risk[i]) if at_risk else None  # only meaningful on at-risk minutes
            summary = None
            if alarm:
                summary = (
                    f"Zone {req.zone}: temperature is at risk of exceeding {model.breach_c:.0f} C "
                    f"within {model.horizon} minutes."
                )
            elif res.state[i] == "breached":
                summary = f"Zone {req.zone}: temperature is above {model.breach_c:.0f} C now."
            points.append(
                BreachPoint(
                    ts=res.ts[i].to_pydatetime(),
                    state=res.state[i],
                    at_risk=at_risk,
                    risk_score=risk,
                    alarm=alarm,
                    summary=summary,
                )
            )
            outcome = "alarm" if alarm else ("no_alarm" if at_risk else "not_at_risk")
            metrics.predictions.labels(model.name, outcome).inc()
        if points[-1].risk_score is not None:
            metrics.breach_risk.observe(points[-1].risk_score)
        return BreachResponse(
            zone=req.zone,
            model=ModelRef(name=model.name, version=model.version),
            threshold=model.threshold,
            min_run_minutes=model.min_run,
            horizon_min=model.horizon,
            breach_c=model.breach_c,
            points=points,
        )

    return app
