# ml-api: model serving

FastAPI service that serves three models: freezer **anomaly detection**, **temperature-breach
early warning**, and an **incident-report classifier**. Code: `services/ml-api/`. All are trained on
**SYNTHETIC** data (see [data-sheet.md](data-sheet.md)); their limits are in the
[anomaly](model-cards/anomaly.md), [breach](model-cards/breach.md) and
[incident](model-cards/incident-nlp.md) model cards.

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| POST | `/v1/anomaly/score` | Is this zone behaving abnormally right now? |
| POST | `/v1/breach/predict` | Is this zone at risk of exceeding -18 C within 30 minutes? |
| POST | `/v1/incidents/classify` | Which kind of incident is this free-text report (damage, temperature, delay, safety)? |
| GET | `/v1/models` | Served model versions, thresholds and the metrics they were registered with |
| GET | `/healthz`, `/readyz` | Liveness; readiness (models loaded) |
| GET | `/metrics` | Prometheus |

OpenAPI docs at `/docs`.

### Request

The caller sends a trailing window of **consecutive 1-minute readings** (oldest first):

```json
{ "zone": "A", "return_last": 1,
  "readings": [ {"ts": "2025-01-07T03:55:00Z", "temp_c": -21.4,
                 "compressor_on": false, "compressor_current_a": 0.3}, "..." ] }
```

`temp_c` is `null` when the sensor dropped out; gaps must be sent as `null`, not skipped.
Minimum history: **400** readings for anomaly, **1500** for breach (plus `return_last - 1`).
Bad windows get HTTP 422 with `{"error": {"code": ..., "message": ...}}`
(`insufficient_history`, `irregular_timestamps`).

### Real response (70 minutes into a power cut that began at 02:45)

```json
{ "zone": "A", "model": {"name": "coldchain-anomaly", "version": "1"},
  "threshold": 5.44, "min_run_minutes": 5,
  "points": [ { "ts": "2025-01-07T03:55:00Z", "score": 4.09, "alarm": false,
                "drivers": {"temperature": 4.09, "compressor current": -1.27}, "summary": null } ] }
```

This is **not** an alarm yet. That is the documented behaviour, not a bug: power cuts are the
anomaly detector's weak spot (median detection delay about 76 minutes; see the model card). Once
the score crosses the threshold for 5 consecutive minutes, `alarm` is `true` and `summary` is a
plain-English sentence for the dashboard. `drivers` give the signed robust z-score per signal (temperature, compressor current and the
compressor-off duration, which is a z-equivalent of how long the compressor has been off; the summary then
states the off time in minutes).

`/v1/breach/predict` returns `state` (`normal`, `recovering`, `breached`, `no_reading`),
`at_risk`, `risk_score`, `alarm`. `risk_score` is a model output in 0..1, **not a calibrated
probability**, and is `null` when the minute is not at risk (already breached or recovering).

### Incident classification

```json
POST /v1/incidents/classify   {"texts": ["Forklift reversed into plt 4412 at dock 3, bottom cartons crushed and wrap torn.",
                                         "asdf qwerty"]}
```

Real output (model `coldchain-incident` v1, threshold `review_below` 0.96):

| Report | label | confidence | needs_review |
|---|---|---|---|
| Forklift reversed into plt 4412 ... cartons crushed | damage | 1.00 | false |
| Freezer 2 alarm since 02:15, probe reading -12C ... | temperature | 1.00 | false |
| the door was left open and a pallet got dented | temperature | 0.99 | false |
| asdf qwerty | delay | 0.64 | **true** |

Notes: 1 to 64 reports per request, each 1 to 2000 characters (else HTTP 422). `confidence` is a
softmax score, **not a calibrated probability**. `needs_review` is true below the threshold; on held-out
synthetic data it flags about a quarter of reports and catches 55-62% of errors, but on the validation
families it was no better than chance (see the [model card](model-cards/incident-nlp.md)). The mixed
report in the third row is answered with 0.99 confidence although it names two issues: the model is
over-confident on mixed reports. The model is **optional**: if it cannot be loaded the endpoint answers
HTTP 503 `model_unavailable` while the other endpoints keep working.

## Design decisions (and why)

- **Stateless scoring.** The caller owns the history and sends a window; the service holds no
  state and needs no database, so it scales by adding replicas. The future `core` service owns
  history in Postgres.
- **One feature definition for training and serving.** The service imports the training code
  (`coldchain_ml.anomaly.features`, `breach.data`, the detectors and the debounce), so serving
  cannot drift from what was evaluated. Tests prove it:
  - `ml/tests/test_serving_parity.py`: features from the trailing window are identical to
    features from the full history (60 random minutes, including fault periods).
  - `test_serving_score_equals_batch_evaluation_score`: the HTTP score equals the batch score.
  - Run-length features are capped at 1,440 min so a bounded window can reproduce them. The cap
    never binds on the simulator's data (longest run: 460 min; checked on all 11 histories), so
    no result changed; a constructed 50-hour run proves the cap is needed in principle.
- **Models come from the MLflow registry** (`models:/<name>@champion`). Roll out or roll back by
  moving the alias; no redeploy of code. The service **fails fast** at startup if a model cannot be
  loaded, so it never reports ready without models.
- **Gated promotion.** `python -m coldchain_ml.registry.promote` refits each production model and
  refuses to register it unless it reproduces the metrics in `docs/results/*.json`. Both
  models reproduced exactly (anomaly recall/precision to 1e-6, breach PR-AUC 0.5712 = 0.5712).
- **Incident model served without PyTorch.** The service runs the int8 ONNX model with onnxruntime
  and the `tokenizers` library (the same `IncidentModel` class the registry gate runs, kept in
  `coldchain_ml.nlp.serving`), so the image carries no PyTorch.
- **Each report is scored on its own.** Dynamic int8 quantisation derives activation scales from the
  whole batch, so batching changes predictions (measured: about 2.7% of reports get a different
  label between a batch of 1 and a batch of 64; fp32 has none). The service loops over the request's
  texts one by one; a regression test fails if a future change batches them again. The same text
  returns the identical answer alone or among others (checked against the real model).
- **The incident gate caught a real problem.** `python -m coldchain_ml.registry.promote_incident`
  refuses to register unless the serving path (onnxruntime plus the `tokenizers` library) reproduces
  the benchmarked macro-F1. The first attempt failed (0.7556 vs 0.7514): the tokenizers were identical
  (520 of 520 texts), the cause was the batch dependence above. After switching to one report at a time it
  reproduced exactly (test 0.7408, independent 0.7950) and `coldchain-incident` v1 was registered.
- **Metrics for ML, not just HTTP:** request count and latency per route, inference time per
  model, predictions by outcome, rejected inputs by reason, loaded model versions, and the score
  distributions of served predictions (`anomaly_score`, `breach_risk_score`, `incident_confidence`)
  as the raw material for the drift monitor.

## Performance (measured)

Machine: Intel i7-13700H laptop, Windows 11, Python 3.12, one uvicorn worker, client on the same
host. 200 requests on real 1-minute windows from a simulator history, after warm-up.
Reproduce: `scripts/bench_ml_api.py`.

| Endpoint | Body | p50 | p95 | p99 | Throughput |
|---|---|---|---|---|---|
| `/v1/anomaly/score` (500 readings) | 62 KB | 36 ms | 56 ms | 66 ms | 24 req/s |
| `/v1/breach/predict` (1500 readings) | 185 KB | 98 ms | 109 ms | 115 ms | 9.6 req/s |

- **It does not scale with threads.** With 4 concurrent clients throughput stays the same (24 and
  9.6 req/s) and latency quadruples: the work is pandas/NumPy under the GIL. Scale with processes
  or replicas. The platform needs 4 zones x 1 score/minute = about 0.07 req/s per site, so one worker has
  large headroom; the limit matters for many sites or backfills.
- **Where the time goes** (in-process, 1500 readings): request parsing 5 ms, window to DataFrame
  6 ms (was 32 ms before I replaced per-reading `model_dump`), features about 20 ms, LightGBM
  and the breach frame the rest.
- **Incident endpoint** (real model, same machine, one run, a loaded laptop): one report per request
  p50 28.5 ms, p95 33.4 ms; 60 reports in one request 1.19 s (about 20 ms each, sequential by design).
  Model-only latency of the int8 model is 21 ms on one thread; see the model card for the full ONNX
  benchmark. This is a single measurement, not a load test.
- **Benchmark pitfalls on Windows:** my first numbers were about 2.7 s per request, including
  `/healthz`. That was client-side (`localhost` resolving to IPv6 first, and a new httpx client
  building an SSL context on every call), not the service. The script now uses `127.0.0.1` and a
  persistent client, and says so in its docstring.

## Run it

```bash
# register the models (writes to ./mlruns, or set MLFLOW_TRACKING_URI)
uv run python -m coldchain_ml.registry.promote

# the incident model needs the NLP artefacts, which are built locally (PyTorch group, ~1 GB):
# see the reproduce steps in docs/model-cards/incident-nlp.md, ending with
uv run python -m coldchain_ml.registry.promote_incident

# serve on http://127.0.0.1:8081
uv run uvicorn coldchain_ml_api.app:create_app --factory --port 8081
```

With Docker: `docker compose --profile core --profile ml up -d --build`, then run `promote` with
`MLFLOW_TRACKING_URI=http://localhost:5000` so the registry inside Compose has a champion.
**Verified:** the image builds in CI (GitHub Actions), imports its dependencies and runs as a
non-root user; and in the Compose stack ([docker-run.md](docker-run.md)) the container loaded the anomaly and
breach models from the Compose MLflow server and scored a live stream. **Not verified:** the incident model in
Compose (registering it needs the opt-in PyTorch group).

## Limitations

- Authentication is one shared key: set `API_KEY` and `/v1/*` requires it (`X-API-Key` or `Authorization: Bearer`);
  `/healthz`, `/readyz` and `/metrics` stay open. No users, rotation or rate limiting, and the key is only as safe
  as the channel (use TLS beyond localhost).
- The caller must provide at least 1500 minutes of history for breach scoring (about 185 KB per
  request). A stateful variant (service keeps a rolling buffer fed from NATS) would cut that;
  it is a trade-off for later, not a missing feature.
- One zone per request; no cross-zone batching.
- Models are the synthetic-data models; thresholds are tuned for F1 on a simulator.
- The incident model's weights are not in the repository (`models/` is git-ignored). A fresh clone
  serves anomaly and breach after `promote`; the incident endpoint stays at 503 until the NLP
  artefacts are rebuilt (17 to 30 minutes of CPU training plus export).
- Incident classification is connected to `core` (`POST /v1/incidents` classifies and stores a report; see
  [core.md](core.md)), but nothing alerts on a label.
- Incident scoring is sequential: 60 reports take over a second.
