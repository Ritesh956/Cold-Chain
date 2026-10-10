# ColdChain AI

An AI control tower for a simulated automated cold store: computer vision on dock cameras,
freezer anomaly detection and breach prediction, volume forecasting, slotting, incident
text classification, and an LLM assistant that answers from the platform's own data.

> **Status: early build.** Simulator, data plane and the first model (freezer anomaly detection)
> are done. Other models are not built yet. The results table holds measured numbers only.

Inspired by public descriptions of highly automated frozen warehouses. It is not affiliated
with, and does not use data from, any real operator. All sensor data is **synthetic** (see
[docs/data-sheet.md](docs/data-sheet.md)).

Plan and architecture: [docs/PROPOSAL.md](docs/PROPOSAL.md).

## Quick start

```bash
uv sync --all-packages          # install the workspace
uv run pytest                   # simulator + schema tests
uv run python -m coldchain_simulator generate --out data/synthetic/freezer
cp .env.example .env
docker compose --profile core up -d --build   # NATS, Postgres/Timescale, Redis, MinIO, MLflow
```

MLflow: http://localhost:5000 · MinIO console: http://localhost:9001 · NATS monitor: http://localhost:8222

## Repo layout

```
libs/coldchain_schemas/   shared event schemas (NATS payloads, ground-truth labels)
services/simulator/       synthetic freezer sensors with labelled faults (offline + live NATS)
ml/                       training and evaluation code (from week 2)
infra/                    compose helpers, MLflow image
docs/                     proposal, data sheet, model cards (as models land)
```

## Results

Every row has a baseline, a stated split and a model card. Rows are added as models are evaluated.

| Task | Data | Baseline | Best simple method | Key result (test) | Card |
|---|---|---|---|---|---|
| Freezer anomaly detection | **Synthetic** (176 test events) | z-score: event-F1 0.93, drift caught after ~3 days | Robust z-score on temperature + compressor current | Same F1 (0.92), but drift caught in **~6 h (12x earlier)**. Isolation forest did not beat it. | [card](docs/model-cards/anomaly.md) |

_Synthetic-data results measure fit to my simulator, not real-world performance._
