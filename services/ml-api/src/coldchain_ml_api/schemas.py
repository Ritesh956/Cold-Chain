"""Request/response contracts. The caller sends a trailing window of 1-minute readings."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from coldchain_ml.contract import MAX_HISTORY
from pydantic import BaseModel, Field


class Reading(BaseModel):
    ts: datetime = Field(description="Minute of the reading; naive timestamps are read as UTC")
    temp_c: float | None = Field(
        default=None, ge=-100, le=100, allow_inf_nan=False, description="null when the sensor dropped out"
    )
    compressor_on: bool
    compressor_current_a: float = Field(ge=0, le=1000, allow_inf_nan=False)


class WindowRequest(BaseModel):
    zone: str = Field(min_length=1, max_length=32)
    readings: list[Reading] = Field(
        max_length=MAX_HISTORY,
        description="Consecutive 1-minute readings, oldest first. Gaps must be sent as temp_c=null.",
    )
    return_last: int = Field(default=1, ge=1, le=1440, description="How many trailing minutes to score")


class ModelRef(BaseModel):
    name: str
    version: str


class AnomalyPoint(BaseModel):
    ts: datetime
    score: float | None = Field(description="Max robust z-score across signals; null if not computable")
    alarm: bool
    drivers: dict[str, float | None] = Field(description="Signed robust z-score per signal")
    summary: str | None = None


class AnomalyResponse(BaseModel):
    zone: str
    model: ModelRef
    threshold: float
    min_run_minutes: int
    points: list[AnomalyPoint]


class BreachPoint(BaseModel):
    ts: datetime
    state: Literal["normal", "recovering", "breached", "no_reading"]
    at_risk: bool
    risk_score: float | None = Field(
        description="Model output in 0..1. A ranking score, NOT a calibrated probability."
    )
    alarm: bool
    summary: str | None = None


class BreachResponse(BaseModel):
    zone: str
    model: ModelRef
    threshold: float
    min_run_minutes: int
    horizon_min: int
    breach_c: float
    points: list[BreachPoint]
