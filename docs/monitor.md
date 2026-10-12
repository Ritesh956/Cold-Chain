# monitor: drift checks, Prometheus, Grafana

`services/monitor/`. Answers one question on a schedule: **is what the models see now still like what
they were built on, and are their outputs still behaving the way they did at deployment?** All data
here is the **SYNTHETIC** freezer stream.

```
core  /v1/zones/{zone}/readings  ──►  input drift  (PSI per zone and feature, last 24 h vs reference)  ─┐
ml-api /metrics (score histograms) ─►  score drift  (PSI of recent scores vs early-deployment baseline) ─┼─► /metrics ─► Prometheus ─► Grafana
                                                                                                          └─► /v1/drift (JSON)
```

## What it measures

| Check | Compared | Reference |
|---|---|---|
| Input drift | `temp_c`, `compressor_current_a`, `compressor_on` over the last `DRIFT_WINDOW_MIN` (1,440) minutes, per zone | Quantile bins of the synthetic training readings, shipped as `reference.json` |
| Score drift | `anomaly_score` and `breach_risk_score` histograms exported by the ml-api | The model's **own** first `SCORE_BASELINE_COUNT` (500) scored minutes after the monitor started |

The statistic is PSI (population stability index). Defaults: below 0.10 stable, 0.10 to 0.25 warn,
0.25 and above drift. Those cut-offs are a convention and are settings (`DRIFT_WARN_PSI`, `DRIFT_PSI`).

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/v1/drift` | The last report: per (zone, feature) PSI, sample count and state; per model score PSI and state |
| POST | `/v1/drift/check` | Run a check now |
| GET | `/healthz`, `/metrics` | Liveness; Prometheus (`drift_input_psi`, `drift_input_flag`, `drift_score_psi`, `drift_score_flag`, ...) |

## Design decisions

- **No verdict without enough data.** Fewer than `DRIFT_MIN_SAMPLES` (200) readings gives
  `insufficient_data`, with no PSI exported. A score baseline that is still filling reports
  `building_baseline`. A quiet monitor is therefore never evidence of stability.
- **Sensor dropouts are skipped, not zero-filled**, so a flaky sensor does not look like a temperature shift.
- **A failing source is an error in the report and a counter, not a crash**, and one failing zone does not
  hide the others.
- **Score drift survives restarts of the ml-api.** A counter that goes backwards, or buckets that change,
  restarts the baseline instead of producing a negative or meaningless difference.
- **Reads the services' public interfaces** (core's REST API, the ml-api's `/metrics`); it has no
  database access and no shared code with them.

## Measured: does it separate normal days from drift?

[docs/results/drift_demo.md](results/drift_demo.md), from `scripts/drift_demo.py`, on the synthetic
history (360 zone-days):

- On days with no injected anomaly, **1 of 272 (0.4%)** were flagged.
- On days **with** an injected anomaly (door left open, compressor drift, power cut), **46 of 88 (52%)**
  were flagged. That is the check working as defined (the inputs differ from the reference), but it
  means this monitor is not an anomaly detector and a flagged day is not necessarily a model problem.
- Injected sensor bias of +0.5 C, compressor current +10%, and a compressor stuck on were each flagged on
  **272 of 272** anomaly-free days.

Caveats: the synthetic noise is low and the injected shifts are uniform, so this is an easy case; the
reference was built from the same history it is tested on; and daily PSI on 1-minute readings is
autocorrelated, so the 0.25 cut-off is a tuned convention, not a calibrated false-alarm rate. Nothing here
says how it behaves on real freezer data.

## Limitations

- Score drift is relative to early deployment, not to validation. If the models were already wrong
  at deployment, it cannot tell.
- No labels, so it measures **input and output shift, not accuracy**. Concept drift that leaves both
  unchanged (for example, a changed definition of a breach) is invisible to it.
- No alert routing: drift is visible in Grafana and `/v1/drift`; it does not page anyone. (Alerts from `core` can
  go to a webhook; drift flags cannot.)
- `API_KEY` protects `/v1/*` (`CORE_API_KEY` and `ML_API_KEY` are what it sends to the services); `/metrics`
  stays open for Prometheus.
- **Run in Docker once** ([docker-run.md](docker-run.md)): Prometheus scraped the monitor, `core` and the ml-api,
  and the Grafana dashboard rendered. **Early verdicts are unreliable:** with under 12 hours of readings per zone
  it gave one `drift` and four `warn` on anomaly-free data; with a full 1,440-minute window none and five `warn`.
  `DRIFT_MIN_SAMPLES` (200) allows a verdict long before a full day (a defrost cycle is 8 hours), so read
  verdicts from the first day with care.

## Run it

```bash
docker compose --profile core --profile ml --profile app --profile sim --profile observability up -d --build
# Grafana http://localhost:3001 (anonymous viewer; admin/admin for local development)
# Monitor http://localhost:8090/v1/drift
```

Without Docker: `uv run uvicorn coldchain_monitor.app:app_factory --factory --port 8090` with `CORE_URL`
and `ML_API_URL` set. Rebuild the reference with
`uv run python -m coldchain_monitor.reference --readings data/synthetic/freezer/readings.parquet`.
