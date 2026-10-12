"""Shared API-key check for the ColdChain services (a pure ASGI middleware: no framework import).

Protects every path under `protect_prefix` (default `/v1`). Health, readiness and metrics stay open so
orchestrators and Prometheus keep working. With no key configured the middleware lets everything through
(local development) and the service should say so at startup: see `describe`.

The key travels in `X-API-Key` or `Authorization: Bearer`. It is compared in constant time. This is a
shared secret between services and one operator console, not per-user authentication: it has no users,
roles, rotation or rate limiting, and without TLS in front of it the key is visible on the wire.
"""

from __future__ import annotations

import hmac
import json
from collections.abc import Awaitable, Callable, MutableMapping
from typing import Any

Scope = MutableMapping[str, Any]
Receive = Callable[[], Awaitable[MutableMapping[str, Any]]]
Send = Callable[[MutableMapping[str, Any]], Awaitable[None]]

_BODY = json.dumps({"error": {"code": "unauthorized", "message": "missing or invalid API key"}}).encode()


def presented_key(headers: list[tuple[bytes, bytes]]) -> str | None:
    """The key a request carries, from X-API-Key or an Authorization: Bearer header."""
    found: str | None = None
    for name, value in headers:
        n = name.lower()
        if n == b"x-api-key":
            return value.decode("latin-1")
        if n == b"authorization":
            scheme, _, token = value.decode("latin-1").partition(" ")
            if scheme.lower() == "bearer" and token:
                found = token.strip()
    return found


def key_matches(presented: str | None, expected: str) -> bool:
    return presented is not None and hmac.compare_digest(presented.encode(), expected.encode())


def describe(key: str | None) -> str:
    return "API key required on /v1" if key else "API auth DISABLED (no API_KEY set): development only"


class ApiKeyMiddleware:
    def __init__(
        self, app: Callable[..., Awaitable[None]], key: str | None, protect_prefix: str = "/v1"
    ) -> None:
        self.app, self.key, self.prefix = app, (key or None), protect_prefix

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        open_ = scope["type"] not in ("http", "websocket") or not self._protected(scope["path"])
        if self.key is None or open_:
            await self.app(scope, receive, send)
            return
        if scope["type"] == "http" and scope.get("method") == "OPTIONS":
            await self.app(scope, receive, send)  # CORS preflight carries no credentials
            return
        if key_matches(presented_key(list(scope.get("headers", []))), self.key):
            await self.app(scope, receive, send)
            return
        if scope["type"] == "websocket":
            await send({"type": "websocket.close", "code": 1008})
            return
        await send(
            {
                "type": "http.response.start",
                "status": 401,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(_BODY)).encode()),
                    (b"www-authenticate", b"Bearer"),
                ],
            }
        )
        await send({"type": "http.response.body", "body": _BODY})

    def _protected(self, path: str) -> bool:
        return path == self.prefix or path.startswith(self.prefix + "/")
