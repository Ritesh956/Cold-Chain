from __future__ import annotations

import os
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from coldchain_core.mlclient import MlApiError
from coldchain_schemas import SensorReading

T0 = datetime(2025, 1, 1, tzinfo=UTC)
MIN = timedelta(minutes=1)


def reading(
    zone: str = "A", i: int = 0, temp: float | None = -25.0, on: bool = True, cur: float = 12.0
) -> SensorReading:
    return SensorReading(ts=T0 + i * MIN, zone=zone, temp_c=temp, compressor_on=on, compressor_current_a=cur)


def history(zone: str, n: int, start: int = 0, temp: float = -25.0) -> list[SensorReading]:
    return [reading(zone, start + i, temp=temp) for i in range(n)]


class FakeMl:
    """Stands in for the ml-api. `alarm_when(ts, kind)` decides alarms; calls are recorded."""

    def __init__(self, anomaly_min: int = 10, breach_min: int = 20, alarm_when=None) -> None:
        self.mins = {"coldchain-anomaly": anomaly_min, "coldchain-breach": breach_min}
        self.alarm_when = alarm_when or (lambda ts, kind: False)
        self.calls: list[tuple[str, str, int, int]] = []  # (kind, zone, n_readings, return_last)
        self.fail: MlApiError | None = None

    async def model_info(self) -> dict[str, Any]:
        return {k: {"version": "1", "min_history_min": v} for k, v in self.mins.items()}

    def _points(self, kind: str, readings: list[dict], return_last: int) -> list[dict]:
        out = []
        for r in readings[-return_last:]:
            ts = datetime.fromisoformat(r["ts"])
            alarm = bool(self.alarm_when(ts, kind))
            if kind == "anomaly":
                out.append(
                    {
                        "ts": r["ts"],
                        "score": 9.0 if alarm else 1.0,
                        "alarm": alarm,
                        "drivers": {},
                        "summary": "unusual" if alarm else None,
                    }
                )
            else:
                breached = bool(self.alarm_when(ts, "breached"))
                out.append(
                    {
                        "ts": r["ts"],
                        "state": "breached" if breached else "normal",
                        "at_risk": not breached,
                        "risk_score": 0.9 if alarm else 0.01,
                        "alarm": alarm,
                        "summary": "at risk" if alarm else None,
                    }
                )
        return out

    async def anomaly(self, zone: str, readings: list[dict], return_last: int) -> dict[str, Any]:
        self.calls.append(("anomaly", zone, len(readings), return_last))
        if self.fail:
            raise self.fail
        return {
            "zone": zone,
            "model": {"name": "coldchain-anomaly", "version": "1"},
            "points": self._points("anomaly", readings, return_last),
        }

    async def breach(self, zone: str, readings: list[dict], return_last: int) -> dict[str, Any]:
        self.calls.append(("breach", zone, len(readings), return_last))
        if self.fail:
            raise self.fail
        return {
            "zone": zone,
            "model": {"name": "coldchain-breach", "version": "1"},
            "points": self._points("breach", readings, return_last),
        }

    async def close(self) -> None:
        return None


class FakeMsg:
    def __init__(self, data: bytes, log: list[str]) -> None:
        self.data, self._log = data, log

    async def ack(self) -> None:
        self._log.append("ack")

    async def nak(self) -> None:
        self._log.append("nak")

    async def term(self) -> None:
        self._log.append("term")


class FakeSource:
    def __init__(self, batches: list[list[bytes]], log: list[str]) -> None:
        self._batches, self._log = list(batches), log

    async def fetch(self, batch: int, timeout: float) -> list[FakeMsg]:
        return [FakeMsg(d, self._log) for d in (self._batches.pop(0) if self._batches else [])]


@pytest.fixture()
def pg_url() -> str | None:
    return os.getenv("TEST_DATABASE_URL") or os.getenv("DATABASE_URL")
