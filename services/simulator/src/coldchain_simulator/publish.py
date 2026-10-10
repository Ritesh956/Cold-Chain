"""Live mode: replay the simulated history onto NATS at an accelerated clock.

Only the readings are published; ground-truth events never leave the simulator.
"""

from __future__ import annotations

import asyncio
import math
from collections.abc import Iterator

from coldchain_schemas import SensorReading, subject_for

from coldchain_simulator.config import SimConfig
from coldchain_simulator.generate import generate


def iter_readings(cfg: SimConfig) -> Iterator[list[SensorReading]]:
    """Yield one list of readings (one per zone) per simulated minute."""
    readings, _ = generate(cfg)
    for ts, grp in readings.groupby("ts", sort=True):
        yield [
            SensorReading(
                ts=ts.to_pydatetime(),
                zone=str(r.zone),
                temp_c=None if math.isnan(r.temp_c) else float(r.temp_c),
                compressor_on=bool(r.compressor_on),
                compressor_current_a=float(r.compressor_current_a),
            )
            for r in grp.itertuples()
        ]


async def run(cfg: SimConfig, nats_url: str, speedup: float) -> None:
    import nats  # imported lazily: only live mode needs the dependency

    nc = await nats.connect(nats_url)
    js = nc.jetstream()
    await js.add_stream(name="SENSORS", subjects=["sensor.>"], max_age=7 * 24 * 3600)
    try:
        for batch in iter_readings(cfg):
            for reading in batch:
                await js.publish(subject_for(reading), reading.model_dump_json().encode())
            await asyncio.sleep(60.0 / speedup)
    finally:
        await nc.drain()
