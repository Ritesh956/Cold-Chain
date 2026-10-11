"""Build the 1-minute-grid window the ml-api expects from whatever readings are stored."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime, timedelta

from coldchain_schemas import SensorReading

MINUTE = timedelta(minutes=1)


def build_window(
    rows: Sequence[SensorReading], end: datetime, minutes: int, min_coverage: float
) -> list[dict] | None:
    """Return `minutes` consecutive readings ending at `end` (inclusive), or None if too sparse.

    A minute with no stored reading is sent as temp_c=null. Compressor state and current are
    carried forward from the previous reading: a missing message is not evidence the compressor
    stopped, and zero-filling would corrupt the duty-cycle features. Returns None when fewer than
    `min_coverage` of the minutes have a real reading, or when the minute at `end` itself is missing.
    """
    start = end - (minutes - 1) * MINUTE
    by_ts = {r.ts: r for r in rows if start <= r.ts <= end}
    if end not in by_ts or len(by_ts) / minutes < min_coverage:
        return None

    prior = [r for r in rows if r.ts < start]
    last = max(prior, key=lambda r: r.ts) if prior else None
    out: list[dict] = []
    for i in range(minutes):
        ts = start + i * MINUTE
        r = by_ts.get(ts)
        if r is not None:
            last = r
            out.append(_row(ts, r.temp_c, r.compressor_on, r.compressor_current_a))
        elif last is not None:
            out.append(_row(ts, None, last.compressor_on, last.compressor_current_a))
        else:  # gap before the first reading we have: nothing to carry forward yet
            first = min(by_ts.values(), key=lambda x: x.ts)
            out.append(_row(ts, None, first.compressor_on, first.compressor_current_a))
    return out


def _row(ts: datetime, temp: float | None, on: bool, current: float) -> dict:
    return {"ts": ts.isoformat(), "temp_c": temp, "compressor_on": on, "compressor_current_a": current}
