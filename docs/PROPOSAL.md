# ColdChain AI — Approved Architecture & Plan

Status: **approved** (2026-10-10). Constraint added at approval: **total spend = ₹0**.

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
