"""Contract check: core's scorer against the REAL ml-api and REAL registered models.

The unit and e2e tests use a fake ml-api; this proves the two services agree on the wire format
and that a real fault produces a real alert. Not hermetic (needs the registered models), so it is
a script, not a pytest test.

    uv run python -m coldchain_ml.registry.promote   # once: register the models
    uv run uvicorn coldchain_ml_api.app:create_app --factory --port 8081
    uv run python scripts/smoke_core_with_ml_api.py
"""

from __future__ import annotations

import asyncio
from datetime import timedelta
from pathlib import Path

import pandas as pd
from coldchain_core.alerts import AlertEngine
from coldchain_core.config import Settings
from coldchain_core.events import BroadcastAlertPublisher, Broadcaster
from coldchain_core.metrics import Metrics
from coldchain_core.mlclient import HttpMlApi
from coldchain_core.repo import MemoryRepo
from coldchain_core.scoring import Scorer
from coldchain_schemas import SensorReading

HISTORY = Path("data/synthetic/anomaly_eval/seed102")
MIN = timedelta(minutes=1)


async def main() -> None:
    readings = pd.read_parquet(HISTORY / "readings.parquet")
    events = pd.read_parquet(HISTORY / "events.parquet")
    readings = readings[readings["zone"] == "A"].reset_index(drop=True)

    anomalies = events[(events["zone"] == "A") & events["is_anomaly"]].sort_values("start")
    door = anomalies[anomalies["type"] == "door_left_open"].iloc[0]
    print(f"real fault: door_left_open {door.start} -> {door.end} ({(door.end - door.start) / MIN:.0f} min)")

    end_i = int(readings["ts"].searchsorted(door.start)) + 90
    start_i = end_i - 3500  # enough history for the breach model's 1500-minute window
    rows = [
        SensorReading(
            ts=r.ts.to_pydatetime(),
            zone="A",
            temp_c=None if pd.isna(r.temp_c) else float(r.temp_c),
            compressor_on=bool(r.compressor_on),
            compressor_current_a=float(r.compressor_current_a),
        )
        for r in readings.iloc[start_i : end_i + 1].itertuples()
    ]

    repo, bcast, metrics = MemoryRepo(), Broadcaster(), Metrics()
    await repo.upsert_readings(rows)
    # begin scoring 40 minutes before the fault and catch up through it
    begin = (door.start - 40 * MIN).to_pydatetime()
    for model in ("anomaly", "breach"):
        await repo.set_scoring_state("A", model, begin)

    settings = Settings(max_chunk_min=60)
    ml = HttpMlApi("http://127.0.0.1:8081")
    scorer = Scorer(
        repo,
        ml,
        AlertEngine(settings.alert_close_after_min),
        BroadcastAlertPublisher(bcast),
        metrics,
        settings,
    )
    async with bcast.subscribe() as q:
        await scorer.start()
        print("ml-api min history:", scorer.min_history)
        for _ in range(4):
            await scorer.tick()
        feed = [q.get_nowait() for _ in range(q.qsize())]
    await ml.close()

    print(f"\nalerts raised ({len(feed)} events):")
    for e in feed:
        a = e["alert"]
        print(
            f"  {e['action']:6s} {a['kind']:12s} {a['severity']:8s} opened {a['opened_at']} "
            f"closed {a['closed_at']} peak {a['peak_score']} | {a['summary']}"
        )
    for model in ("anomaly", "breach"):
        print(f"scoring position {model}: {await repo.get_scoring_state('A', model)}")
    print(
        "metrics:",
        {
            f"{m}/{o}": metrics.scored.labels(m, o)._value.get()
            for m in ("anomaly", "breach")
            for o in ("ok", "ml_error_retry", "ml_error_rejected", "skipped_insufficient_data")
            if metrics.scored.labels(m, o)._value.get()
        },
    )


if __name__ == "__main__":
    asyncio.run(main())
