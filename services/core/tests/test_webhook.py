from __future__ import annotations

import asyncio
import hashlib
import hmac
import json

import httpx
from coldchain_core.alerts import AlertEvent
from coldchain_core.events import WebhookAlertPublisher, alert_text
from coldchain_schemas import Alert, AlertKind
from core_helpers import T0


def event(action="opened", kind=AlertKind.BREACHED, severity="critical", summary="above -18 C") -> AlertEvent:
    a = Alert(id=1, zone="C", kind=kind, severity=severity, opened_at=T0, summary=summary)
    return AlertEvent(action, a)


class Receiver:
    """A fake webhook endpoint: records requests, answers with a scripted sequence of statuses."""

    def __init__(self, statuses=(200,)) -> None:
        self.statuses, self.requests = list(statuses), []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        status = self.statuses.pop(0) if len(self.statuses) > 1 else self.statuses[0]
        if status == "boom":
            raise httpx.ConnectError("refused")
        return httpx.Response(status)


def make(receiver, **kw) -> WebhookAlertPublisher:
    client = httpx.AsyncClient(transport=httpx.MockTransport(receiver))
    return WebhookAlertPublisher("http://hook.test/x", client=client, backoff_s=0.0, **kw)


async def run(pub, *events):
    await pub.start()
    for e in events:
        await pub.publish(e)
    await pub.drain()
    await pub.aclose()


async def test_delivers_json_with_ready_to_show_text():
    rx = Receiver()
    pub = make(rx)
    await run(pub, event())
    (req,) = rx.requests
    body = json.loads(req.content)
    assert body["type"] == "alert" and body["action"] == "opened" and body["alert"]["zone"] == "C"
    assert body["text"] == "[CRITICAL] breached in zone C: opened. above -18 C"
    assert req.headers["content-type"] == "application/json" and "x-coldchain-signature" not in req.headers
    assert (pub.delivered, pub.failed, pub.dropped) == (1, 0, 0)


async def test_signature_lets_the_receiver_verify_the_body():
    rx = Receiver()
    pub = make(rx, secret="s3cret")
    await run(pub, event())
    (req,) = rx.requests
    want = "sha256=" + hmac.new(b"s3cret", req.content, hashlib.sha256).hexdigest()
    assert req.headers["x-coldchain-signature"] == want


async def test_only_configured_actions_are_sent():
    rx = Receiver()
    pub = make(rx)  # default: opened and closed
    await run(pub, event("opened"), event("acknowledged"), event("closed"))
    assert [json.loads(r.content)["action"] for r in rx.requests] == ["opened", "closed"]
    rx2 = Receiver()
    await run(make(rx2, actions=frozenset({"acknowledged"})), event("opened"), event("acknowledged"))
    assert [json.loads(r.content)["action"] for r in rx2.requests] == ["acknowledged"]


async def test_retries_server_errors_and_transport_failures_then_succeeds():
    rx = Receiver([503, "boom", 200])
    pub = make(rx)
    await run(pub, event())
    assert len(rx.requests) == 3 and (pub.delivered, pub.failed) == (1, 0)


async def test_gives_up_after_the_attempt_limit_and_counts_the_failure():
    rx = Receiver([500])
    pub = make(rx)
    await run(pub, event())
    assert len(rx.requests) == 3 and (pub.delivered, pub.failed) == (0, 1)


async def test_client_errors_are_not_retried():
    rx = Receiver([404])
    pub = make(rx)
    await run(pub, event())
    assert len(rx.requests) == 1 and (pub.delivered, pub.failed) == (0, 1)


async def test_a_dead_endpoint_never_blocks_the_caller_and_the_queue_drops_the_oldest():
    rx = Receiver([500])
    client = httpx.AsyncClient(transport=httpx.MockTransport(rx))
    pub = WebhookAlertPublisher("http://hook.test/x", client=client, backoff_s=10.0, queue_size=2)
    # the worker is not started: publishing must return immediately and never raise
    await asyncio.wait_for(asyncio.gather(*(pub.publish(event()) for _ in range(5))), 1.0)
    assert pub.dropped == 3
    await client.aclose()


def test_alert_text_without_a_summary():
    assert alert_text(event("closed", AlertKind.BREACH_RISK, "warning", summary=None)) == (
        "[WARNING] breach risk in zone C: closed"
    )
