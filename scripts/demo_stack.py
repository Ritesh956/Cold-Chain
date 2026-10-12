"""Run the real `core` service without Docker, replaying a SYNTHETIC history that contains a real fault.

It uses core's real API, ingest worker, scorer and alert engine, with in-memory storage instead of
Postgres/Redis/NATS, and the REAL ml-api (you start it). Meant for looking at the dashboard.

    uv run python -m coldchain_ml.registry.promote                      # once
    uv run uvicorn coldchain_ml_api.app:create_app --factory --port 8081
    uv run python scripts/demo_stack.py                                 # core on http://127.0.0.1:8000
    cd dashboard && npm run dev                                         # http://localhost:3000

Optional: the drift monitor, pointed at this core and the ml-api:
    CORE_URL=http://127.0.0.1:8000 ML_API_URL=http://127.0.0.1:8081 \
        uv run uvicorn coldchain_monitor.app:app_factory --factory --port 8090
"""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import replace
from datetime import timedelta
from pathlib import Path

import pandas as pd
import uvicorn
from coldchain_core.app import Deps, create_app
from coldchain_core.cache import MemoryCache
from coldchain_core.config import Settings
from coldchain_core.events import (
    BroadcastAlertPublisher,
    Broadcaster,
    CompositePublisher,
    WebhookAlertPublisher,
)
from coldchain_core.mlclient import HttpMlApi
from coldchain_core.repo import MemoryRepo
from coldchain_schemas import SensorReading

HISTORY = Path("data/synthetic/anomaly_eval/seed102")
MIN = timedelta(minutes=1)
PRELOAD_MIN = 3500  # the breach model needs 1500 minutes of history


class _Msg:
    def __init__(self, data: bytes) -> None:
        self.data = data

    async def ack(self) -> None: ...
    async def nak(self) -> None: ...
    async def term(self) -> None: ...


class ReplaySource:
    """Hands the ingest worker one simulated minute of every zone per tick, at `rate` minutes/second."""

    def __init__(self, rows: list[SensorReading], rate: float) -> None:
        by_ts: dict = {}
        for r in rows:
            by_ts.setdefault(r.ts, []).append(r)
        self._steps = [by_ts[t] for t in sorted(by_ts)]
        self._i, self._rate = 0, rate

    async def fetch(self, batch: int, timeout: float) -> list[_Msg]:
        if self._i >= len(self._steps):
            await asyncio.sleep(timeout)
            return []
        await asyncio.sleep(1 / self._rate)
        step = self._steps[self._i]
        self._i += 1
        return [_Msg(r.model_dump_json().encode()) for r in step]


def _readings(df: pd.DataFrame) -> list[SensorReading]:
    return [
        SensorReading(
            ts=r.ts.to_pydatetime(),
            zone=str(r.zone),
            temp_c=None if pd.isna(r.temp_c) else float(r.temp_c),
            compressor_on=bool(r.compressor_on),
            compressor_current_a=float(r.compressor_current_a),
        )
        for r in df.itertuples()
    ]


async def main(rate: float, lead_min: int, port: int, ml_url: str) -> None:
    readings = pd.read_parquet(HISTORY / "readings.parquet")
    events = pd.read_parquet(HISTORY / "events.parquet")
    door = events[(events["type"] == "door_left_open") & events["is_anomaly"]].sort_values("start").iloc[0]
    zone = str(door["zone"])
    print(f"replaying SYNTHETIC history; door_left_open in zone {zone}: {door.start} -> {door.end}")

    start = door.start - lead_min * MIN
    pre = readings[(readings["ts"] >= start - PRELOAD_MIN * MIN) & (readings["ts"] < start)]
    live = readings[(readings["ts"] >= start) & (readings["ts"] <= door.end + 90 * MIN)]

    repo, cache, bcast = MemoryRepo(), MemoryCache(), Broadcaster()
    await repo.upsert_readings(_readings(pre))
    for z in readings["zone"].unique():
        for model in ("anomaly", "breach"):
            await repo.set_scoring_state(str(z), model, start - MIN)

    # API_KEY / ML_API_KEY / ALERT_WEBHOOK_* from the environment apply here too
    settings = replace(Settings.from_env(), score_every_min=5, score_tick_s=1.0)
    ml = HttpMlApi(ml_url, api_key=settings.ml_api_key)
    sinks = [BroadcastAlertPublisher(bcast)]
    closers = [ml.close]
    if settings.alert_webhook_url:
        webhook = WebhookAlertPublisher(
            settings.alert_webhook_url,
            secret=settings.alert_webhook_secret,
            actions=frozenset(settings.alert_webhook_actions.split(",")),
        )
        await webhook.start()
        sinks.append(webhook)
        closers.insert(0, webhook.aclose)
    deps = Deps(
        repo=repo,
        cache=cache,
        ml=ml,
        publisher=CompositePublisher(*sinks),
        broadcaster=bcast,
        source=ReplaySource(_readings(live), rate),
        closers=closers,
    )
    app = create_app(deps, settings)
    await uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")).serve()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rate", type=float, default=3.0, help="simulated minutes per second")
    ap.add_argument("--lead-min", type=int, default=90, help="minutes of replay before the fault starts")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--ml-api", default="http://127.0.0.1:8081")
    a = ap.parse_args()
    asyncio.run(main(a.rate, a.lead_min, a.port, a.ml_api))
