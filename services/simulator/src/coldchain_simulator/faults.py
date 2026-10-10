"""Builds the per-minute forcing signals (door, power, defrost, efficiency) and the
ground-truth event list for one freezer zone."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta

import numpy as np
from coldchain_schemas import FaultEvent, FaultType
from numpy.typing import NDArray

from coldchain_simulator.config import MINUTES_PER_DAY, SimConfig

EDGE_MARGIN_MIN = 2 * MINUTES_PER_DAY  # keep anomalies away from the very start/end


@dataclass(frozen=True)
class Forcings:
    door_frac: NDArray[np.float64]  # 0..1 fraction of the door open each minute
    power_off: NDArray[np.bool_]
    defrost: NDArray[np.bool_]
    efficiency: NDArray[np.float64]  # compressor efficiency, 1.0 = healthy
    events: list[FaultEvent]


def _place(
    rng: np.random.Generator,
    taken: list[tuple[int, int]],
    length: int,
    n_steps: int,
    guard: int,
) -> tuple[int, int]:
    """Pick a random [start, end) that keeps `guard` minutes clear of `taken`."""
    hi = n_steps - EDGE_MARGIN_MIN - length
    if hi <= EDGE_MARGIN_MIN:
        raise ValueError("horizon too short for the requested anomalies")
    for _ in range(2000):
        start = int(rng.integers(EDGE_MARGIN_MIN, hi))
        end = start + length
        if all(end + guard <= s or start - guard >= e for s, e in taken):
            return start, end
    raise ValueError("could not place anomalies without overlap; lower the counts or extend days")


def build_forcings(cfg: SimConfig, zone: str, zone_idx: int, rng: np.random.Generator) -> Forcings:
    n = cfg.n_steps
    door = np.zeros(n)
    power_off = np.zeros(n, dtype=bool)
    defrost = np.zeros(n, dtype=bool)
    eff = np.ones(n)
    events: list[FaultEvent] = []
    taken: list[tuple[int, int]] = []

    def ts(i: int):  # minute index -> datetime
        return cfg.start + timedelta(minutes=i)

    def record(kind: FaultType, s: int, e: int, note: str = "") -> None:
        events.append(
            FaultEvent(zone=zone, type=kind, start=ts(s), end=ts(e), is_anomaly=kind.is_anomaly, note=note)
        )

    # Anomalies: longest first so they are easiest to fit.
    for _ in range(cfg.n_compressor_drift):
        length = int(rng.integers(cfg.drift_days[0], cfg.drift_days[1] + 1)) * MINUTES_PER_DAY
        s, e = _place(rng, taken, length, n, cfg.anomaly_guard_min)
        taken.append((s, e))
        eff[s:e] = np.linspace(1.0, cfg.drift_final_efficiency, e - s)  # repaired at e
        record(FaultType.COMPRESSOR_DRIFT, s, e, "efficiency ramps down, repaired at end")

    for _ in range(cfg.n_power_cut):
        length = int(rng.integers(cfg.power_cut_min[0], cfg.power_cut_min[1] + 1))
        s, e = _place(rng, taken, length, n, cfg.anomaly_guard_min)
        taken.append((s, e))
        power_off[s:e] = True
        record(FaultType.POWER_CUT, s, e)

    for _ in range(cfg.n_door_left_open):
        length = int(rng.integers(cfg.door_open_min[0], cfg.door_open_min[1] + 1))
        s, e = _place(rng, taken, length, n, cfg.anomaly_guard_min)
        taken.append((s, e))
        door[s:e] = 1.0
        record(FaultType.DOOR_LEFT_OPEN, s, e)

    # Benign: periodic defrost, staggered per zone so zones do not defrost together.
    offset = 120 + 60 * zone_idx
    for s in range(offset, n, cfg.defrost_every_min):
        e = min(s + cfg.defrost_len_min, n)
        defrost[s:e] = True
        record(FaultType.DEFROST, s, e)

    # Benign: routine forklift door traffic during working hours.
    minute_of_day = np.arange(n) % MINUTES_PER_DAY
    working = (minute_of_day >= 6 * 60) & (minute_of_day < 22 * 60)
    starts = np.flatnonzero(working & (rng.random(n) < cfg.routine_doors_per_hour / 60.0))
    for s in starts:
        length = int(rng.integers(1, 4))
        e = min(int(s) + length, n)
        door[int(s) : e] = np.maximum(door[int(s) : e], cfg.routine_door_intensity)
        record(FaultType.ROUTINE_DOOR, int(s), e)

    events.sort(key=lambda ev: ev.start)
    return Forcings(door, power_off, defrost, eff, events)
