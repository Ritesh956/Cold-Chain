# ColdChain AI — Approved Architecture & Plan

Status: **approved** (2026-10-10), partly built (see [Progress and deviations](#progress-and-deviations)).
Constraint added at approval: **total spend = ₹0**.

## Zero-cost constraint — how each piece is covered

| Need | Free option used | Notes |
|---|---|---|
| Training GPU | Google Colab / Kaggle free GPU | Laptop (i7-13700H, 15.7 GB RAM, no GPU) used for CPU inference + benchmarks only |
| CI/CD | GitHub Actions (public repo) | Unlimited minutes for public repos |
| Container registry | GHCR | Free for public images |
| Experiment tracking / registry | Self-hosted MLflow (in Compose) | No SaaS |
| Annotation | CVAT (self-host or cvat.ai free tier) or Label Studio | Dataset stays small (~200 images) |
| LLM for assistant | Provider interface: Groq free tier / Gemini free tier / local Ollama | Azure OpenAI is an optional adapter, never required |
| Live demo hosting | **Hugging Face Spaces (free CPU Docker, 2 vCPU / 16 GB)** for the backend; Vercel or Azure Static Web Apps free tier for the frontend | Azure VM only if Azure for Students verifies with **no card** (credit-capped, cannot bill) |
| Datasets | M5 (Kaggle, free token), Roboflow Universe, NAB, OSHA severe-injury reports | Never committed; download scripts only. Licences recorded in `docs/data-sheet.md` |

Rule: no service that requires a credit card on file. If a free tier asks for one, we do not use it.

## Architecture

```
 looped video ─► MediaMTX (RTSP) ─► vision (YOLO-ONNX + ByteTrack + barcode/OCR) ─► MinIO
 simulator (synthetic sensors, ground-truth faults) ─────────────────────────────┐
                                                                                  ▼
                                                           NATS JetStream (sensor.*, vision.*, alerts.*)
                                                                                  │
                                                  core (ingest + REST + SSE) ─► Postgres/Timescale, Redis
                                                                                  │
                          ml-api (forecast, anomaly, breach, slotting, incident NLP) ◄─ MLflow registry
                          assistant (LLM tool-calling over core + ml-api)
                          monitor (PSI/KS drift → Prometheus) ─► Grafana
                          dashboard (Next.js, SSE live updates)
```

Services are split by **runtime profile**, not one-per-model: `ml-api` (light CPU), `vision` (heavy, stateful stream loop), `assistant` (I/O-bound, rate-limited).

## Cuts

Go ingestion (only if a benchmark shows Python is the bottleneck), Detectron2, Kubernetes as primary deploy (Helm optional + k3d smoke test), DVC.
Cut order if time runs out: damage classifier → HDBSCAN → Helm.

## Model plan

| Task | Baseline | Model | Metrics |
|---|---|---|---|
| Volume forecast | Seasonal naive, moving avg | LightGBM global | WAPE, MASE, RMSE; rolling-origin split (no MAPE: unstable on low volume) |
| Slotting | ABC-XYZ rules | k-means / HDBSCAN | Silhouette, stability, agreement with ABC-XYZ |
| Freezer anomaly | z-score, rolling z | Isolation Forest, residual detector | Event-level P/R/F1, detection delay |
| Breach prediction | Persistence | LightGBM | PR-AUC, lead time |
| Detection/tracking | YOLO n vs s | YOLO fine-tune + ByteTrack | mAP50/50-95 (scene-held-out), count error |
| Label reading | — | pyzbar barcode, then OCR | Exact match, CER |
| Damaged pallets (exploratory) | — | EfficientNet/ResNet | F1 + CI |
| Edge inference | ONNX FP32 | INT8 | p50/p95 latency, FPS, mAP drop |
| Incident NLP | TF-IDF + LogReg | DistilBERT → ONNX int8 | Macro-F1; split by generation group |
| Assistant | — | Tool-calling LLM | Tool accuracy, arg accuracy, numeric grounding |

## Honesty rules

- Synthetic data is labelled synthetic everywhere (`docs/data-sheet.md`, README, model cards).
- Synthetic-data results are never presented as real-world performance; real-data checks (NAB, OSHA, hand-written test set) sit beside them.
- Weaknesses go in each model card.
- No NewCold name/branding in the product; described as "inspired by public descriptions of automated cold stores".

## 8-week plan

1. Foundation, simulator, M5 EDA, start collecting frames
2. Forecasting + clustering
3. Anomaly + breach; CVAT labelling; YOLO baseline
4. Detector, tracking/counting, ONNX/INT8, RTSP worker
5. Incident NLP, OCR/barcode, damage clf, `ml-api`, Prometheus
6. Assistant + eval harness, `core`, dashboard v1
7. Drift monitor, Grafana, integration CI, deploy
8. README, GIF, model cards, results, resume bullets, interview Q&A (buffer)

## Progress and deviations

Last updated 2026-10-12 (after the auth, webhook, incident and anomaly follow-ups). The 8-week plan above is a work-package schedule, not elapsed calendar time.

| Plan item | Status | Where |
|---|---|---|
| Repo, CI, simulator, Compose data plane | Done | `services/simulator`, `docker-compose.yml`, `.github/workflows/ci.yml` |
| Freezer anomaly detection (baselines vs models, event-level metrics) | Done, synthetic + a 3-failure real-data check | [model card](model-cards/anomaly.md) |
| Breach early warning | Done, synthetic | [model card](model-cards/breach.md) |
| Model serving + MLflow registry + metrics | Done | [ml-api](ml-api.md) |
| `core` (NATS ingest, Postgres history, alerts, live API) | Done (built ahead of its planned week); since then: incident reports, alert acknowledgement and webhook, shared-key auth, opt-in retention | [core](core.md) |
| Incident NLP (BERT fine-tune, ONNX int8, serving) | Done on synthetic text; **no gain over TF-IDF**; connected to `core` and the dashboard; real-text check done: safety recall 0.89 (0.83 in cold storage) | [model card](model-cards/incident-nlp.md) |
| Forecasting and clustering/slotting | Not started (needs the Kaggle M5 token) | |
| CVAT labelling, YOLO, tracking, ONNX/INT8 for vision, RTSP worker | Not started | |
| OCR/barcode, damaged-pallet classifier | Not started | |
| LLM assistant + eval harness | Not started | |
| Drift monitor, Grafana | Built; compose config validated, **never run** (Docker was off) | [monitor](monitor.md), `services/monitor`, `infra/observability` |
| Dashboard (Next.js) | Built; browser-checked against a no-Docker replay, Docker image not built locally | [dashboard](dashboard.md), `dashboard/` |
| Cloud deployment | Not started; CI builds the images and runs integration tests on real NATS/TimescaleDB/Redis | |
| README, model cards, results | Done for what is built: README with a screen-capture GIF, 3 model cards plus service docs, [resume bullets and interview Q&A](interview-notes.md) | |

Deviations from the plan, and why:

- **Incident text is generated programmatically, not by an LLM.** No free LLM key was available yet.
  The generator is deterministic and tested for leakage; the OSHA real-text check was done later (one-sided, safety recall 0.89; see the model card).
  A 120-report independent set was written by the AI assistant (synthetic, labelled as such).
- **The BERT result is "no measurable gain".** The plan asked for a model that is compared honestly with
  a baseline; it is, and the baseline ties. The transformer is still the served model because the project
  is meant to demonstrate that stack; the model card says so and says when to switch.
- **ml-api serves three models, one as an optional model**, and scores incident reports one at a time
  (dynamic int8 quantisation made batched results depend on their neighbours).
- **Kubernetes/Helm, Go ingestion and Detectron2 remain cut**, as planned.
- **Azure:** nothing deployed. The images build in CI; the free-tier plan above is unchanged.
