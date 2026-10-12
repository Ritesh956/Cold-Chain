from __future__ import annotations

import asyncio

from coldchain_schemas.auth import ApiKeyMiddleware, describe, key_matches, presented_key


def run(mw, path="/v1/x", headers=(), method="GET", scope_type="http"):
    """Drive the middleware once; return (status or None, inner_app_called, sent messages)."""
    called = []
    sent = []

    async def inner(scope, receive, send):
        called.append(scope["path"])
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"ok"})

    async def send(msg):
        sent.append(msg)

    async def receive():
        return {"type": "http.request"}

    app = mw(inner)
    raw = [(k.encode(), v.encode()) for k, v in headers]
    scope = {"type": scope_type, "path": path, "method": method, "headers": raw}
    asyncio.run(app(scope, receive, send))
    status = next((m["status"] for m in sent if m["type"] == "http.response.start"), None)
    return status, bool(called), sent


def guarded(key="s3cret"):
    return lambda inner: ApiKeyMiddleware(inner, key)


def test_missing_key_is_401_and_the_app_is_not_reached():
    status, called, sent = run(guarded())
    assert status == 401 and not called
    assert any(h == (b"www-authenticate", b"Bearer") for h in sent[0]["headers"])
    assert b"unauthorized" in sent[1]["body"]


def test_wrong_key_is_401():
    assert run(guarded(), headers=[("x-api-key", "nope")])[0] == 401
    assert run(guarded(), headers=[("authorization", "Bearer nope")])[0] == 401


def test_both_header_styles_are_accepted():
    assert run(guarded(), headers=[("x-api-key", "s3cret")])[0:2] == (200, True)
    assert run(guarded(), headers=[("authorization", "Bearer s3cret")])[0:2] == (200, True)
    assert run(guarded(), headers=[("Authorization", "bearer s3cret")])[0] == 200  # case-insensitive


def test_a_basic_auth_header_is_not_a_key():
    assert run(guarded(), headers=[("authorization", "Basic czNjcmV0")])[0] == 401


def test_open_paths_never_need_a_key():
    for path in ("/healthz", "/readyz", "/metrics", "/docs", "/v10/x"):
        assert run(guarded(), path=path)[0:2] == (200, True), path


def test_prefix_itself_and_subpaths_are_protected():
    assert run(guarded(), path="/v1")[0] == 401
    assert run(guarded(), path="/v1/zones/A/readings")[0] == 401


def test_no_key_configured_means_no_auth():
    assert run(guarded(None))[0:2] == (200, True)
    assert run(guarded(""))[0:2] == (200, True)


def test_preflight_is_allowed_without_a_key():
    assert run(guarded(), method="OPTIONS")[0] == 200


def test_websocket_without_a_key_is_closed():
    _, called, sent = run(guarded(), scope_type="websocket")
    assert not called and sent == [{"type": "websocket.close", "code": 1008}]


def test_helpers():
    assert presented_key([(b"X-API-Key", b"k")]) == "k"
    assert presented_key([(b"authorization", b"Bearer  tok ")]) == "tok"
    assert presented_key([]) is None
    assert key_matches("a", "a") and not key_matches("a", "b") and not key_matches(None, "a")
    assert "DISABLED" in describe(None) and "required" in describe("k")
