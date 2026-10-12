# core: ingest, history, scoring orchestration, alerts, API

`services/core/`. It connects the pieces: it consumes the **SYNTHETIC** sensor stream from NATS,
keeps the history in Postgres, asks the [ml-api](ml-api.md) to score each zone, turns the results
into alerts, and exposes everything to the dashboard and (later) the assistant.

```
simulator ──► NATS JetStream (SENSORS) ──► ingest worker ──► Postgres/Timescale  (history, alerts, scoring position)
                                              │                  ▲   │
                                              └──► Redis (latest reading per zone)
                                                                 │   ▼
                        ml-api ◄── scorer (window from Postgres, per zone and model) ──► alert engine
                                                                                          │
                          REST + SSE API ◄── broadcaster ◄───────────────────────────────┤
                          NATS ALERTS stream (alerts.<kind>.<zone>) ◄─────────────────────┘
```

## API

| Method | Path | Purpose |
|---|---|---|
| GET | `/v1/zones` | Every zone: latest reading, open alerts, how far each model has scored |
| GET | `/v1/zones/{zone}/readings?minutes=360` | Recent history (1 to 10,080 minutes) |
| GET | `/v1/alerts?status=active\|all&zone=&limit=` | Alert episodes, newest first |
| POST | `/v1/incidents` | Body `{text, zone?}`: classify a free-text incident report with the ml-api, store it, push it to live clients. 503 if the classifier is down (nothing stored), 422 for invalid text |
| GET | `/v1/incidents?label=&needs_review=&limit=` | Stored incident reports, newest first |
| GET | `/v1/stream` | Server-Sent Events: `reading`, `alert` and `incident` events, with keep-alives |
| GET | `/healthz`, `/readyz`, `/metrics` | Liveness; readiness (database, cache, NATS); Prometheus |

Alert kinds: `anomaly` (readings unusual), `breach_risk` (at risk of exceeding -18 C within 30 min),
`breached` (above the limit now, severity `critical`). Each alert is an episode with `opened_at`,
`closed_at`, `peak_score`, a plain-English `summary` and the `model_version` that raised it.

## Design decisions

- **Idempotent at-least-once ingest.** JetStream redelivers, so the insert is
  `ON CONFLICT (zone, ts) DO NOTHING` (first write wins). A message is acknowledged only **after**
  the database commit; a failed write NAKs the batch so nothing is lost. An undecodable message is
  **terminated and counted** instead of being redelivered forever, so one bad message cannot block
  the stream.
- **Scoring position per (zone, model), in event time.** The scorer asks the ml-api for every
  minute since its last position (`return_last=k`), so a short alarm between two ticks is not
  missed. The position moves only after the ml-api answered **and** the alerts were stored, so an
  ml-api outage delays alerts but never silently skips minutes (tested). Windows with too little
  stored history (warm-up, or an outage inside the window) are skipped and counted
  (`skipped_insufficient_data`) instead of retried forever.
- **Gap handling is explicit.** A missing minute is sent as `temp_c = null`, and the compressor
  state and current are carried forward from the previous reading. A missing message is not
  evidence the compressor stopped, and zero-filling would corrupt the duty-cycle features.
- **Alert state machine with hysteresis.** An alert opens on the first alarm minute and closes
  only after 5 consecutive quiet minutes, so a flapping model produces one alert, not many. Open
  alerts are re-adopted from the database after a restart, so a restart cannot duplicate them.
- **Ports and adapters.** The logic runs against a `Repo` contract with an in-memory fake and a
  Postgres implementation. **The same contract test suite runs against both**, which is what
  makes the fake trustworthy in every other test.
- **The database is the source of truth; Redis is an optimisation.** A cache failure is counted
  and ignored; the data is already stored. The cache never lets a replayed older message overwrite
  a newer reading.
- **TimescaleDB if present, plain Postgres otherwise.** `readings` becomes a hypertable when the
  extension exists; the same schema runs on plain Postgres.
- **A slow SSE client loses its oldest events** instead of blocking ingest.
- **Supervised background loops.** A crashed loop is logged and restarted; the loops always yield
  to the event loop (a test caught a version that would have starved it on an instantly-empty
  source).

## How it was tested, and where

| Level | Where it runs | What it proves |
|---|---|---|
| Unit and contract with the in-memory repo | locally and CI | window building, alert lifecycle, ingest ordering/idempotency/poison handling, scorer chunking and outage behaviour, API |
| Live SSE over a real socket | locally and CI | events reach a connected client; disconnect cleans up |
| Repo contract against **real TimescaleDB** | CI `integration` job | the Postgres SQL behaves like the fake |
| End to end through **real NATS JetStream, TimescaleDB, Redis** | CI `integration` job | a simulated 7-day history published to JetStream ends up stored exactly once (including after republishing), the cache is current, and the alerts in Postgres and on the ALERTS stream equal what the alert engine produces offline from the same data |
| `core` against the **real ml-api and registered models** | `scripts/smoke_core_with_ml_api.py` (manual) | the two services agree on the wire format |

Result of the last row, on a real 22-minute door-left-open: breach-risk warning 4 minutes after
the door opened (6 minutes before the temperature crossed -18 C), anomaly alert at minute 5, a
critical `breached` alert at the crossing, and clean closes. The CI integration job uses a **fake**
ml-api (its models are covered by the ml-api tests); only the manual script exercises the real one.

Things CI caught that local tests could not: a TimescaleDB hypertable cannot be dropped in the
same statement as other tables, and a 5-day simulation cannot place the e2e test's anomalies.
Both were bugs in the test setup, not the service.

## Run it

```bash
docker compose --profile core --profile ml --profile app --profile sim up -d --build
# register models first, against the Compose MLflow:
MLFLOW_TRACKING_URI=http://localhost:5000 uv run python -m coldchain_ml.registry.promote
curl http://localhost:8000/v1/zones
curl -N http://localhost:8000/v1/stream
```

The simulator replays at 60x (one simulated minute per second) and `core` scores every 5 event
minutes. **The full Compose stack has not been run end to end** (Docker was not running on the
development machine); CI builds the images and runs the integration job on service containers.

## Limitations

- **No authentication or rate limiting.** Internal service; add both before any exposure.
- **No backfill on first start:** a zone first seen mid-stream is scored from its newest minute.
- **After an outage longer than about 5% of a model's window** (75 minutes for breach), that model
  cannot score until the gap leaves its window (up to about 25 hours for breach), and says so in
  `core_scoring_total{outcome="skipped_insufficient_data"}`.
- **An alert's peak score is persisted only when it opens and closes**; a crash while open loses
  later peak updates.
- **Single scorer and single ingest consumer.** Fine for this scale; multiple replicas would need
  leader election or sharding by zone.
- No data retention policy yet (Timescale retention would be the natural fit).
- **Incident reports are classified and stored, but never raise an alert.** A `safety` label is shown in the
  dashboard, not escalated. The classifier was trained on synthetic text and saw no real reports; the score
  is a ranking signal, not a probability, and `needs_review` marks low-scoring reports. The Postgres side of
  the incident table is covered by the repository contract tests, which run against real Postgres in CI only.
