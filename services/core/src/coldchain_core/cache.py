"""Latest-reading cache (Redis). The database stays the source of truth; this only makes the
dashboard's 'current state of every zone' cheap."""

from __future__ import annotations

from typing import Protocol

from coldchain_schemas import SensorReading


class LatestCache(Protocol):
    async def set_latest(self, reading: SensorReading) -> None: ...
    async def get_latest(self, zone: str) -> SensorReading | None: ...
    async def ping(self) -> bool: ...
    async def close(self) -> None: ...


class MemoryCache:
    def __init__(self) -> None:
        self._d: dict[str, SensorReading] = {}

    async def set_latest(self, reading: SensorReading) -> None:
        cur = self._d.get(reading.zone)
        if cur is None or reading.ts >= cur.ts:
            self._d[reading.zone] = reading

    async def get_latest(self, zone: str) -> SensorReading | None:
        return self._d.get(zone)

    async def ping(self) -> bool:
        return True

    async def close(self) -> None:
        return None


class RedisCache:
    def __init__(self, url: str) -> None:
        import redis.asyncio as redis

        self._r = redis.from_url(url, decode_responses=True)

    @staticmethod
    def _key(zone: str) -> str:
        return f"latest:{zone}"

    async def set_latest(self, reading: SensorReading) -> None:
        # Never let a replayed older message overwrite a newer value. The ingest worker is the
        # only writer, so read-compare-write is sufficient here (no cross-process race).
        cur = await self.get_latest(reading.zone)
        if cur is None or reading.ts >= cur.ts:
            await self._r.set(self._key(reading.zone), reading.model_dump_json())

    async def get_latest(self, zone: str) -> SensorReading | None:
        raw = await self._r.get(self._key(zone))
        return SensorReading.model_validate_json(raw) if raw else None

    async def ping(self) -> bool:
        try:
            return bool(await self._r.ping())
        except Exception:
            return False

    async def close(self) -> None:
        await self._r.aclose()
