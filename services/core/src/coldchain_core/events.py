"""Fan-out of live events to SSE subscribers and to NATS."""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any, Protocol

import httpx
from coldchain_schemas import alert_subject

from coldchain_core.alerts import AlertEvent

log = logging.getLogger("coldchain_core.events")


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


def alert_text(event: AlertEvent) -> str:
    a = event.alert
    kind = a.kind.value.replace("_", " ")
    head = f"[{a.severity.upper()}] {kind} in zone {a.zone}: {event.action}"
    return f"{head}. {a.summary}" if a.summary else head


class WebhookAlertPublisher:
    """POSTs alert events to an operator-configured URL (a chat webhook, an incident tool, a script).

    Delivery happens in a background worker so a slow or dead endpoint can never stall scoring. A bounded
    queue drops the OLDEST pending event when full; a delivery is retried a few times on transport errors
    and 5xx/429, never on other 4xx. Failures are logged and counted in attributes (`delivered`, `failed`,
    `dropped`); this is at-most-once alerting, not a durable outbox: events queued when the process stops
    are lost. The body carries a ready-to-show `text` (Slack-style webhooks display it) plus the full
    alert; with a secret, `X-ColdChain-Signature: sha256=<hmac of the body>` lets the receiver verify it.
    """

    RETRY_STATUS = frozenset({429, 500, 502, 503, 504})

    def __init__(
        self,
        url: str,
        secret: str = "",
        actions: frozenset[str] = frozenset({"opened", "closed"}),
        client: httpx.AsyncClient | None = None,
        attempts: int = 3,
        backoff_s: float = 1.0,
        queue_size: int = 1000,
    ) -> None:
        self.url, self.secret, self.actions = url, secret, actions
        self._client = client or httpx.AsyncClient(timeout=5.0)
        self._owns_client = client is None
        self.attempts, self.backoff_s = attempts, backoff_s
        self._q: asyncio.Queue[AlertEvent] = asyncio.Queue(queue_size)
        self._task: asyncio.Task | None = None
        self.delivered = self.failed = self.dropped = 0

    async def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._run())

    async def aclose(self) -> None:
        if self._task:
            self._task.cancel()
            await asyncio.gather(self._task, return_exceptions=True)
            self._task = None
        if self._owns_client:
            await self._client.aclose()

    async def publish(self, event: AlertEvent) -> None:
        if event.action not in self.actions:
            return
        if self._q.full():
            self._q.get_nowait()
            self.dropped += 1
        self._q.put_nowait(event)

    async def drain(self) -> None:
        """Wait until everything queued so far has been attempted (tests and clean shutdown)."""
        await self._q.join()

    def _body(self, event: AlertEvent) -> bytes:
        payload = alert_payload(event)
        payload["text"] = alert_text(event)
        return json.dumps(payload).encode()

    async def _run(self) -> None:
        while True:
            event = await self._q.get()
            try:
                if await self._deliver(event):
                    self.delivered += 1
                else:
                    self.failed += 1
            except asyncio.CancelledError:
                raise
            except Exception:
                self.failed += 1
                log.exception("alert webhook: unexpected error")
            finally:
                self._q.task_done()

    async def _deliver(self, event: AlertEvent) -> bool:
        body = self._body(event)
        headers = {"content-type": "application/json"}
        if self.secret:
            digest = hmac.new(self.secret.encode(), body, hashlib.sha256).hexdigest()
            headers["X-ColdChain-Signature"] = f"sha256={digest}"
        for attempt in range(1, self.attempts + 1):
            try:
                r = await self._client.post(self.url, content=body, headers=headers)
                if r.status_code < 300:
                    return True
                if r.status_code not in self.RETRY_STATUS:
                    log.warning("alert webhook rejected the event (HTTP %s); not retrying", r.status_code)
                    return False
                reason = f"HTTP {r.status_code}"
            except httpx.HTTPError as e:
                reason = repr(e)
            if attempt < self.attempts:
                await asyncio.sleep(self.backoff_s * 2 ** (attempt - 1))
            else:
                log.warning("alert webhook gave up after %d attempts: %s", attempt, reason)
        return False


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
