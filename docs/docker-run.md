# The full stack, run in Docker

First end-to-end run of the Compose stack, on 2026-10-13: Docker Desktop 29.6.1 on Windows 11 (the Docker VM
had 7.6 GiB of memory), profiles `core`, `ml`, `app`, `observability` and `sim` together. All data is the
**SYNTHETIC** freezer stream. This page says what that run showed and what it did not cover; CI (which builds
the images and runs the integration tests, but never starts the stack) is described in the README.

## What was verified

| Layer | Result |
|---|---|
| Images | `core`, `ml-api`, `dashboard`, `monitor` and `mlflow` all build. |
| Data plane | NATS, Redis and TimescaleDB 2.17.2 (PostgreSQL 16) healthy; `readings` is a real hypertable (2 chunks, 14,332 rows after about 2.5 simulated days). |
| MLflow | Serves on `/health`; artifacts on a Docker volume. The anomaly and breach models were registered from the host with `coldchain_ml.registry.promote` (the refit metrics equal the reported ones exactly: recall 0.9773, precision 0.9305, breach PR-AUC 0.5712, warned >= 5 min 0.7234) and the ml-api container loaded both as `champion` v1. |
| Pipeline | Simulator -> NATS -> `core` -> ml-api -> alerts -> dashboard, running at 60x: every zone scored every few simulated minutes. |
| Alerts | The stream's first real fault (a door left open in zone D, 10:24 to 10:59) produced a breach-risk warning at 10:27, an anomaly alert at 10:36 and a `breached` critical alert at 10:38: **the warning came 11 minutes before the temperature crossed -18 C**. All three closed as the zone recovered. Acknowledging an alert through the API worked and was idempotent. |
| Dashboard | Docker-built image served on the host; zones, the Live indicator, the critical state, both alert texts and the fault region on the chart rendered correctly in the in-app browser. |
| Observability | Prometheus scraped `core`, `ml-api` and `monitor` (all up). The provisioned Grafana dashboard rendered with real series for the first time. The monitor reported input drift (per zone and feature) and score drift for both models. |
| Incident classifier | Registered afterwards (after restoring the opt-in PyTorch group): the gate reproduced the reported macro-F1 exactly (0.7408 test, 0.7950 independent), and after an ml-api restart the container served it. Four hand-written sample reports were labelled as expected (a smoke test, not an accuracy measure). A report posted to `core` was classified, stored in TimescaleDB and listed; an empty text got 422; a report typed into the dashboard form was classified (`damage`, zone C), saved and appeared in the live list without a reload. |
| Authentication | In a second pass with `API_KEY` and `DASHBOARD_PASSWORD` set (random test values): `core`, the ml-api and the monitor answered 401 to a missing or wrong key on `/v1/*` and 200 to `X-API-Key` or `Authorization: Bearer`; `/healthz` and `/metrics` stayed open and Prometheus kept scraping; `core` kept scoring through the keyed ml-api and the monitor read both of its sources with keys; the dashboard refused a missing or wrong password (401) and served with the right one. |
| Webhook | A local receiver got an `opened` event for zone B's compressor-drift alert from inside the `core` container, with a valid `X-ColdChain-Signature` (HMAC-SHA256 of the body). Only `opened` was seen; `closed` and retries were not exercised. |
| Retention | With `RETENTION_DAYS=3`, `core` purged at startup: 39,832 readings became 17,612; the oldest remaining reading was Jan 4 22:05 against a newest of Jan 7 23:27 a minute or so later (the cutoff is 3 days before the newest reading at purge time). Alerts were kept. |
| Restart recovery | Recreating `core` while an anomaly alert was open: the alert was adopted, not duplicated (8 alerts before and after, one open). `core` logged 46 `ml-api unreachable` errors while the ml-api was still starting, then scored normally without a manual restart. |
| Forecast | The history-only outflow forecaster was registered (`coldchain-forecast` v1, after its gate reproduced the reported WAPE exactly), the rebuilt ml-api container served it, and a request with real M5 histories got sensible 28-day forecasts: p50 26 ms for 1 item, 51 ms for 50, 96 ms for 200 (7 requests each). Plumbing and latency only. |
| Footprint | About 1.6 GiB of memory across the 11 containers while running (MLflow 525 MiB, ml-api 330 MiB, Grafana 193 MiB, the rest under 135 MiB each). |

## What the run found (and what changed)

- **MinIO could not be pulled, so the stack did not start.** The `minio/minio` and `minio/mc` images are no longer
  on Docker Hub or Quay. CI never noticed because it does not start the stack. MinIO and its init job were removed;
  MLflow now keeps artifacts on a Docker volume (`--serve-artifacts`). Add an S3-compatible store back when the vision
  track needs object storage.
- **Host ports can clash.** Another local project already held 8080, 3000, 3001 and 9090. The ml-api, dashboard,
  Prometheus and Grafana ports are now `ML_API_PORT`, `DASHBOARD_PORT`, `PROMETHEUS_PORT` and `GRAFANA_PORT`
  in `.env` (defaults unchanged).
- **A Windows console crashed the model registration.** MLflow prints an emoji line when it talks to a server over
  HTTP; a cp1252 console cannot encode it, and the script died after the model was already registered. Fixed in
  `coldchain_ml.common.tracking`.
- **Breach prediction needs 25 simulated hours of history** (`BREACH_MIN_HISTORY` = 1,500 minutes; the anomaly model
  needs 400). After a first start, or after a gap in the stream, breach scoring reports `skipped_insufficient_data`
  for that long: about 25 minutes at the simulator's 60x, a full day in real time.
- **`last_scored` means "scored up to", not "last successful score".** Skipped minutes also advance it, so a zone
  can show a recent `last_scored` for a model that has not produced a score yet.
- **The drift monitor is unreliable on partial windows.** With under 12 hours of readings per zone it gave
  a `drift` verdict on 1 of 12 anomaly-free zone-feature pairs and `warn` on 4; with a full 1,440-minute window it gave
  none and `warn` on 5 (compressor current in all four zones, plus zone A temperature). The minimum sample count
  (200) allows a verdict long before a full day, and a defrost cycle is 8 hours. Treat early verdicts with caution.
  Score drift was `ok` (PSI 0.048 anomaly, 0.014 breach).
- **Registering the incident model needs the opt-in PyTorch group** (`uv sync --all-packages --group nlp`, about
  1 GB), because `promote_incident` imports the training code for one constant. A plain `uv sync` removes that
  group again. The ml-api container itself does not need it.
- **While the ml-api restarted, the monitor said so**: its report listed `GET /metrics failed` instead of
  crashing or going silent, and the dashboard's drift panel showed that line.
- **`docker compose up` recreates services when `.env` changes**, including Postgres and MLflow (their data
  volumes survive, so nothing was lost), and the ml-api briefly drops out, which `core` rides out by retrying.
- **A second excursion in zone D at 13:00 to 13:10.** The zone had barely recovered from the door fault (-19.8 C at
  13:00); the scheduled defrost then pushed it back over the limit (peak -16.6 C). The breach-risk warning came 8
  minutes before that crossing, so the breach alerts match the data. The anomaly alert opened 2 minutes into the
  defrost, which the labels treat as benign; one stressed zone does not show whether that is a false alarm.

## Not verified

- **Incident classification quality in Docker:** only the plumbing was exercised (see above), not accuracy; the
  model card has the measured results.
- Restarts beyond the one above: stopping and starting the data plane mid-stream, the ml-api failing fast when the
  registry has no champion, a long soak. Webhook `closed` events and retry behaviour. Authentication with a browser's
  own password prompt (checked with HTTP requests only).
- Anything beyond one fault: one door-open fault and one defrost excursion say nothing about detection rates (those
  are in the model cards).
- Grafana panel contents beyond "renders with data"; browsers other than the in-app one.
