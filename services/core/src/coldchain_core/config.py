from __future__ import annotations

import os
from dataclasses import dataclass

from coldchain_core.retention import validate as validate_retention


@dataclass(frozen=True)
class Settings:
    nats_url: str = "nats://localhost:4222"
    database_url: str = "postgresql://coldchain:coldchain_dev@localhost:5432/coldchain"
    redis_url: str = "redis://localhost:6379/0"
    ml_api_url: str = "http://127.0.0.1:8081"
    api_key: str = ""  # protects /v1/*; empty disables auth (development only)
    ml_api_key: str = ""  # sent to the ml-api
    retention_days: int = 0  # 0 keeps every reading; otherwise at least 3 (see retention.py)
    retention_every_s: float = 3600.0
    alert_webhook_url: str = ""  # empty: no webhook
    alert_webhook_secret: str = ""  # signs the body (HMAC-SHA256) when set
    alert_webhook_actions: str = "opened,closed"  # any of opened, closed, acknowledged

    # scoring cadence is in EVENT minutes (reading timestamps), not wall-clock: when the simulator
    # replays at 60x, one wall-clock second is one event minute.
    score_every_min: int = 1
    score_tick_s: float = 2.0  # how often the scorer wakes up (wall clock)
    max_chunk_min: int = 60  # minutes scored per ml-api call
    min_coverage: float = 0.95  # share of window minutes that must have a stored reading
    alert_close_after_min: int = 5  # consecutive quiet minutes before an alert closes
    ingest_batch: int = 200
    ingest_wait_s: float = 1.0

    @classmethod
    def from_env(cls) -> Settings:
        d = cls()

        def f(name: str, default, cast=str):
            raw = os.getenv(name)
            return default if raw is None else cast(raw)

        return cls(
            nats_url=f("NATS_URL", d.nats_url),
            database_url=f("DATABASE_URL", d.database_url),
            redis_url=f("REDIS_URL", d.redis_url),
            ml_api_url=f("ML_API_URL", d.ml_api_url),
            api_key=f("API_KEY", d.api_key),
            ml_api_key=f("ML_API_KEY", d.ml_api_key),
            retention_days=validate_retention(f("RETENTION_DAYS", d.retention_days, int)),
            retention_every_s=f("RETENTION_EVERY_S", d.retention_every_s, float),
            alert_webhook_url=f("ALERT_WEBHOOK_URL", d.alert_webhook_url),
            alert_webhook_secret=f("ALERT_WEBHOOK_SECRET", d.alert_webhook_secret),
            alert_webhook_actions=f("ALERT_WEBHOOK_ACTIONS", d.alert_webhook_actions),
            score_every_min=f("SCORE_EVERY_MIN", d.score_every_min, int),
            score_tick_s=f("SCORE_TICK_S", d.score_tick_s, float),
            max_chunk_min=f("MAX_CHUNK_MIN", d.max_chunk_min, int),
            min_coverage=f("MIN_COVERAGE", d.min_coverage, float),
            alert_close_after_min=f("ALERT_CLOSE_AFTER_MIN", d.alert_close_after_min, int),
            ingest_batch=f("INGEST_BATCH", d.ingest_batch, int),
            ingest_wait_s=f("INGEST_WAIT_S", d.ingest_wait_s, float),
        )
