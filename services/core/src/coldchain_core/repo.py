"""Storage port + in-memory implementation.

`Repo` is the contract. `MemoryRepo` (tests, local dev) and `PgRepo` (coldchain_core.pg) must
behave identically; tests/test_repo_contract.py runs the same suite against both.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime, timedelta
from typing import Protocol

from coldchain_schemas import Alert, SensorReading


class Repo(Protocol):
    async def init(self) -> None: ...
    async def close(self) -> None: ...
    async def ping(self) -> bool: ...

    async def upsert_readings(self, rows: Sequence[SensorReading]) -> int:
        """Insert readings; an existing (zone, ts) is left untouched. Returns rows newly inserted."""

    async def zones(self) -> list[str]: ...
    async def latest(self, zone: str) -> SensorReading | None: ...

    async def window(self, zone: str, end: datetime, minutes: int) -> list[SensorReading]:
        """Readings with end-(minutes-1) <= ts <= end, ascending, plus at most one reading just
        before that range (so gap filling can carry the compressor state forward)."""

    async def get_scoring_state(self, zone: str, model: str) -> datetime | None: ...
    async def set_scoring_state(self, zone: str, model: str, ts: datetime) -> None: ...

    async def save_alert(self, alert: Alert) -> Alert:
        """Insert (id is None) or update (by id). Returns the stored alert with its id."""

    async def open_alerts(self) -> list[Alert]: ...
    async def list_alerts(self, active_only: bool, zone: str | None, limit: int) -> list[Alert]:
        """Newest first by opened_at."""


class MemoryRepo:
    def __init__(self) -> None:
        self._readings: dict[str, dict[datetime, SensorReading]] = {}
        self._state: dict[tuple[str, str], datetime] = {}
        self._alerts: dict[int, Alert] = {}
        self._next_id = 1

    async def init(self) -> None:
        return None

    async def close(self) -> None:
        return None

    async def ping(self) -> bool:
        return True

    async def upsert_readings(self, rows: Sequence[SensorReading]) -> int:
        new = 0
        for r in rows:
            zone = self._readings.setdefault(r.zone, {})
            if r.ts not in zone:
                zone[r.ts] = r
                new += 1
        return new

    async def zones(self) -> list[str]:
        return sorted(self._readings)

    async def latest(self, zone: str) -> SensorReading | None:
        z = self._readings.get(zone)
        return z[max(z)] if z else None

    async def window(self, zone: str, end: datetime, minutes: int) -> list[SensorReading]:
        start = end - timedelta(minutes=minutes - 1)
        z = self._readings.get(zone, {})
        inside = sorted((r for r in z.values() if start <= r.ts <= end), key=lambda r: r.ts)
        before = [r for r in z.values() if r.ts < start]
        return ([max(before, key=lambda r: r.ts)] if before else []) + inside

    async def get_scoring_state(self, zone: str, model: str) -> datetime | None:
        return self._state.get((zone, model))

    async def set_scoring_state(self, zone: str, model: str, ts: datetime) -> None:
        self._state[(zone, model)] = ts

    async def save_alert(self, alert: Alert) -> Alert:
        if alert.id is None:
            alert = alert.model_copy(update={"id": self._next_id})
            self._next_id += 1
        self._alerts[alert.id] = alert
        return alert

    async def open_alerts(self) -> list[Alert]:
        return [a for a in self._alerts.values() if a.closed_at is None]

    async def list_alerts(self, active_only: bool, zone: str | None, limit: int) -> list[Alert]:
        out = [
            a
            for a in self._alerts.values()
            if (not active_only or a.closed_at is None) and (zone is None or a.zone == zone)
        ]
        return sorted(out, key=lambda a: a.opened_at, reverse=True)[:limit]
