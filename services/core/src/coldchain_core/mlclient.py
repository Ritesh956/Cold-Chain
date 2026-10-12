"""Client for the ml-api."""

from __future__ import annotations

from typing import Any, Protocol

import httpx


class MlApiError(Exception):
    """`retryable` is True for transport/5xx errors, False for 4xx (the request itself was refused)."""

    def __init__(self, message: str, retryable: bool) -> None:
        super().__init__(message)
        self.retryable = retryable


class MlApi(Protocol):
    async def model_info(self) -> dict[str, Any]: ...
    async def anomaly(self, zone: str, readings: list[dict], return_last: int) -> dict[str, Any]: ...
    async def breach(self, zone: str, readings: list[dict], return_last: int) -> dict[str, Any]: ...
    async def classify(self, texts: list[str]) -> dict[str, Any]: ...
    async def close(self) -> None: ...


class HttpMlApi:
    def __init__(self, base_url: str, timeout_s: float = 15.0, api_key: str = "") -> None:
        headers = {"X-API-Key": api_key} if api_key else {}
        self._c = httpx.AsyncClient(base_url=base_url, timeout=timeout_s, headers=headers)

    async def _call(self, method: str, path: str, **kw: Any) -> dict[str, Any]:
        try:
            r = await self._c.request(method, path, **kw)
        except httpx.HTTPError as e:
            raise MlApiError(f"ml-api unreachable: {e!r}", retryable=True) from e
        if r.status_code >= 500:
            raise MlApiError(f"ml-api {r.status_code}", retryable=True)
        if r.status_code >= 400:
            raise MlApiError(f"ml-api rejected request: {r.status_code} {r.text[:200]}", retryable=False)
        return r.json()

    async def model_info(self) -> dict[str, Any]:
        return await self._call("GET", "/v1/models")

    async def anomaly(self, zone: str, readings: list[dict], return_last: int) -> dict[str, Any]:
        body = {"zone": zone, "readings": readings, "return_last": return_last}
        return await self._call("POST", "/v1/anomaly/score", json=body)

    async def breach(self, zone: str, readings: list[dict], return_last: int) -> dict[str, Any]:
        body = {"zone": zone, "readings": readings, "return_last": return_last}
        return await self._call("POST", "/v1/breach/predict", json=body)

    async def classify(self, texts: list[str]) -> dict[str, Any]:
        return await self._call("POST", "/v1/incidents/classify", json={"texts": texts})

    async def close(self) -> None:
        await self._c.aclose()
