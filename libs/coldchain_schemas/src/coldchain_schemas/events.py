"""Event payloads exchanged over NATS and stored as ground truth.

All sensor data in this project is SYNTHETIC (see docs/data-sheet.md).
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, Field


class FaultType(StrEnum):
    """Injected events. `is_anomaly` marks which ones a detector SHOULD flag."""

    DOOR_LEFT_OPEN = "door_left_open"  # anomaly
    COMPRESSOR_DRIFT = "compressor_drift"  # anomaly
    POWER_CUT = "power_cut"  # anomaly
    DEFROST = "defrost"  # benign, periodic: a hard negative
    ROUTINE_DOOR = "routine_door"  # benign, short forklift traffic: a hard negative

    @property
    def is_anomaly(self) -> bool:
        return self in {
            FaultType.DOOR_LEFT_OPEN,
            FaultType.COMPRESSOR_DRIFT,
            FaultType.POWER_CUT,
        }


class SensorReading(BaseModel):
    """One 1-minute reading for one freezer zone. `temp_c` is None on sensor dropout."""

    ts: datetime
    zone: str
    temp_c: float | None
    compressor_on: bool
    compressor_current_a: float


class FaultEvent(BaseModel):
    """Ground-truth label for an injected event (never sent to detectors)."""

    zone: str
    type: FaultType
    start: datetime
    end: datetime
    is_anomaly: bool
    note: str = Field(default="")


def subject_for(reading: SensorReading) -> str:
    """NATS subject convention: sensor.temp.<zone>."""
    return f"sensor.temp.{reading.zone.lower()}"
