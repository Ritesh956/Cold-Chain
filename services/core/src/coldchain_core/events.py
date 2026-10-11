"""Fan-out of live events to SSE subscribers and to NATS."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any, Protocol

from coldchain_schemas import alert_subject

from coldchain_core.alerts import AlertEvent


class Broadcaster:
    """In-process pub/sub. A slow subscriber loses its OLDEST events instead of blocking ingest."""

    def __init__(self, queue_size: int = 256) -> None:
        self._subs: set[asyncio.Queue[dict[str, Any]]] = set()
        self._size = queue_size
        self.dropped = 0

    @asynccontextmanager
    async def subscribe(self) -> AsyncIterator[asyncio.Queue[dict[str, Any]]]:
        q: asyncio.Queue[dict[str, Any]] = asyncio.Queue(self._size)
        self._subs.add(q)
        try:
            yield q
        finally:
            self._subs.discard(q)

    @property
    def subscribers(self) -> int:
        return len(self._subs)

    def publish(self, event: dict[str, Any]) -> None:
        for q in self._subs:
            if q.full():
                q.get_nowait()
                self.dropped += 1
            q.put_nowait(event)


class AlertPublisher(Protocol):
    async def publish(self, event: AlertEvent) -> None: ...


def alert_payload(event: AlertEvent) -> dict[str, Any]:
    return {"type": "alert", "action": event.action, "alert": json.loads(event.alert.model_dump_json())}


class BroadcastAlertPublisher:
    def __init__(self, broadcaster: Broadcaster) -> None:
        self._b = broadcaster

    async def publish(self, event: AlertEvent) -> None:
        self._b.publish(alert_payload(event))


class NatsAlertPublisher:
    """Publishes alert events to JetStream subject alerts.<kind>.<zone>."""

    def __init__(self, js) -> None:
        self._js = js

    async def publish(self, event: AlertEvent) -> None:
        a = event.alert
        await self._js.publish(alert_subject(a.kind.value, a.zone), json.dumps(alert_payload(event)).encode())


class CompositePublisher:
    """Publishes to every sink; one failing sink must not stop the others."""

    def __init__(self, *sinks: AlertPublisher, on_error=None) -> None:
        self._sinks, self._on_error = sinks, on_error

    async def publish(self, event: AlertEvent) -> None:
        for s in self._sinks:
            try:
                await s.publish(event)
            except Exception as e:
                if self._on_error:
                    self._on_error(s, e)
