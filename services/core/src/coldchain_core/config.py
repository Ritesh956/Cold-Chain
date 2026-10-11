from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    nats_url: str = "nats://localhost:4222"
    database_url: str = "postgresql://coldchain:coldchain_dev@localhost:5432/coldchain"
    redis_url: str = "redis://localhost:6379/0"
    ml_api_url: str = "http://127.0.0.1:8081"

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
            score_every_min=f("SCORE_EVERY_MIN", d.score_every_min, int),
            score_tick_s=f("SCORE_TICK_S", d.score_tick_s, float),
            max_chunk_min=f("MAX_CHUNK_MIN", d.max_chunk_min, int),
            min_coverage=f("MIN_COVERAGE", d.min_coverage, float),
            alert_close_after_min=f("ALERT_CLOSE_AFTER_MIN", d.alert_close_after_min, int),
            ingest_batch=f("INGEST_BATCH", d.ingest_batch, int),
            ingest_wait_s=f("INGEST_WAIT_S", d.ingest_wait_s, float),
        )
