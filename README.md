# ColdChain AI

An AI control tower for a simulated automated cold store: computer vision on dock cameras,
freezer anomaly detection and breach prediction, volume forecasting, slotting, incident
text classification, and an LLM assistant that answers from the platform's own data.

> **Status: week 1 of 8.** Foundation and synthetic data are done. Models are not built yet,
> so there is no results table yet. The table will be filled only with measured numbers.

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

_Filled in as models are evaluated. Each row will have a baseline, a split description and
its model card._
