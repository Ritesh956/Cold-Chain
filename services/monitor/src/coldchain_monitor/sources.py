from __future__ import annotations

from typing import Protocol

import httpx


class SourceError(Exception):
    pass


class DataSource(Protocol):
    async def zones(self) -> list[str]: ...
    async def readings(self, zone: str, minutes: int) -> list[dict]: ...
    async def ml_metrics(self) -> str: ...
    async def close(self) -> None: ...


class HttpSource:
    """Reads recent readings from the core API and metrics text from the ml-api."""

    def __init__(self, core_url: str, ml_api_url: str, timeout_s: float = 15.0) -> None:
        self._core = httpx.AsyncClient(base_url=core_url, timeout=timeout_s)
        self._ml = httpx.AsyncClient(base_url=ml_api_url, timeout=timeout_s)

    async def _get(self, client: httpx.AsyncClient, path: str, **kw) -> httpx.Response:
        try:
            r = await client.get(path, **kw)
            r.raise_for_status()
        except httpx.HTTPError as e:
            raise SourceError(f"GET {path} failed: {e!r}") from e
        return r

    async def zones(self) -> list[str]:
        return [z["zone"] for z in (await self._get(self._core, "/v1/zones")).json()]

    async def readings(self, zone: str, minutes: int) -> list[dict]:
        r = await self._get(self._core, f"/v1/zones/{zone}/readings", params={"minutes": minutes})
        return r.json()["readings"]

    async def ml_metrics(self) -> str:
        return (await self._get(self._ml, "/metrics")).text

    async def close(self) -> None:
        await self._core.aclose()
        await self._ml.aclose()
