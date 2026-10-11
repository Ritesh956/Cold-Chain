"""Consume sensor readings from NATS JetStream into the database.

Delivery is at-least-once, so the write is idempotent (a replayed (zone, ts) is ignored) and a
message is acknowledged only AFTER the database commit. An undecodable message is terminated and
counted instead of being redelivered forever.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Protocol

from coldchain_schemas import SensorReading
from coldchain_schemas.streams import SENSOR_STREAM, SENSOR_SUBJECTS
from pydantic import ValidationError

from coldchain_core.cache import LatestCache
from coldchain_core.events import Broadcaster
from coldchain_core.metrics import Metrics
from coldchain_core.repo import Repo

log = logging.getLogger("coldchain_core.ingest")


class Message(Protocol):
    data: bytes

    async def ack(self) -> None: ...
    async def nak(self) -> None: ...
    async def term(self) -> None: ...


class Source(Protocol):
    async def fetch(self, batch: int, timeout: float) -> list[Message]: ...


class NatsSource:
    """Durable pull consumer on the SENSORS stream."""

    def __init__(self, js, durable: str = "core-ingest") -> None:
        self._js, self._durable, self._sub = js, durable, None

    async def fetch(self, batch: int, timeout: float) -> list[Message]:
        from nats.errors import TimeoutError as NatsTimeout

        if self._sub is None:
            self._sub = await self._js.pull_subscribe(
                SENSOR_SUBJECTS[0], durable=self._durable, stream=SENSOR_STREAM
            )
        try:
            return await self._sub.fetch(batch, timeout=timeout)
        except NatsTimeout:
            return []


class IngestWorker:
    def __init__(
        self,
        source: Source,
        repo: Repo,
        cache: LatestCache,
        broadcaster: Broadcaster,
        metrics: Metrics,
        batch: int = 200,
        wait_s: float = 1.0,
    ) -> None:
        self.source, self.repo, self.cache = source, repo, cache
        self.broadcaster, self.metrics = broadcaster, metrics
        self.batch, self.wait_s = batch, wait_s

    async def process_once(self) -> int:
        """Handle one batch. Returns the number of messages received."""
        msgs = await self.source.fetch(self.batch, self.wait_s)
        valid: list[tuple[Message, SensorReading]] = []
        for m in msgs:
            try:
                valid.append((m, SensorReading.model_validate_json(m.data)))
            except ValidationError:
                self.metrics.poison.inc()
                await m.term()
        if not valid:
            return len(msgs)

        readings = [r for _, r in valid]
        try:
            inserted = await self.repo.upsert_readings(readings)
        except Exception:
            self.metrics.ingest_errors.labels("database").inc()
            for m, _ in valid:
                await m.nak()  # redeliver; nothing was committed
            raise
        for m, _ in valid:
            await m.ack()

        self.metrics.ingested.labels("new").inc(inserted)
        self.metrics.ingested.labels("duplicate").inc(len(readings) - inserted)
        self.metrics.last_ingest_unixtime.set(time.time())

        newest: dict[str, SensorReading] = {}
        for r in readings:
            if r.zone not in newest or r.ts > newest[r.zone].ts:
                newest[r.zone] = r
        for r in newest.values():
            try:
                await self.cache.set_latest(r)
            except Exception:
                # the cache is an optimisation; the data is already safely stored
                self.metrics.ingest_errors.labels("cache").inc()
            self.broadcaster.publish({"type": "reading", "reading": r.model_dump(mode="json")})
        return len(msgs)

    async def run(self, backoff_s: float = 2.0) -> None:
        while True:
            try:
                await self.process_once()
                await asyncio.sleep(0)  # always yield: an instantly-empty source must not starve the loop
            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception("ingest batch failed; retrying in %.1fs", backoff_s)
                await asyncio.sleep(backoff_s)
