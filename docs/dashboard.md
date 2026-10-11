# dashboard: live zones, alerts, drift

`dashboard/`. Next.js (App Router) + React, no UI or chart library. It shows every freezer zone live,
the alert list and the drift monitor's verdicts. **All data is SYNTHETIC.**

```
browser ──► Next.js server ──/api/core/*────► core     (REST history + SSE /v1/stream)
                          └─/api/monitor/*──► monitor  (/v1/drift, polled every 30 s)
```

The browser only talks to the Next.js server, which proxies to the services, so neither needs CORS.
The proxy targets (`CORE_URL`, `MONITOR_URL`) are fixed **at build time**.

## What it shows

- **Zone cards:** latest temperature and compressor state, a temperature chart with the -18 C limit and
  shaded spans for alert episodes, and the open alerts with the model's plain-English summary.
  Health is written out as well as coloured (Normal / Warning / Critical / No data).
- **Alerts:** newest first, with open/closed state.
- **Model drift:** PSI per model score and per (zone, feature), with the monitor's state, including
  "not enough data" and "building baseline" instead of a false "stable".
- **Connection state** (Live / Reconnecting / Offline). The page reseeds from REST after every reconnect,
  so a dropped stream leaves no gap in the charts.

## Design decisions

- **Stream first, then history.** The SSE connection opens before the history loads; duplicates are
  dropped because only strictly newer readings extend a series.
- **Events are batched** (500 ms) before rendering. At the simulator's 60x replay four zones produce
  about 240 events per second; rendering each one would saturate the page.
- **State logic is a pure module** (`lib/state.ts`) with unit tests (17, vitest): dedupe, ordering,
  capping, alert upsert, history/live merge, health, downsampling.
- **Sensor dropouts stay as gaps** in the line, not zeros.

## Run it

Without Docker (replays a real synthetic fault through the real `core` logic, ml-api and scorer, with
in-memory storage; see the docstring of `scripts/demo_stack.py`):

```bash
uv run python -m coldchain_ml.registry.promote
uv run uvicorn coldchain_ml_api.app:create_app --factory --port 8081
uv run python scripts/demo_stack.py            # core on :8000
cd dashboard && npm ci && npm run dev          # http://localhost:3000
```

With Compose: the `app` profile includes the dashboard on http://localhost:3000.

## Verified, and not

- **Verified:** unit tests; type check; production build; and a manual run in a browser against
  `scripts/demo_stack.py` with the real ml-api: live updates, alerts (a critical breach and warnings
  raised by the real models), the drift table, no console errors, no horizontal scroll at 375 px.
- **Not verified:** the Docker image (Docker was not running; CI builds it) and the Compose stack with the
  dashboard in it; browsers other than the in-app one; screen readers (it uses landmarks, `role=status`
  and text labels, but was not tested with one).

## Limitations

- No authentication. The proxy exposes core and monitor to whoever can reach the dashboard.
- Shows about 12 h of history per zone (720 readings); no date picker, no zoom.
- Read-only: it cannot acknowledge or close alerts.
- `npm audit` reports advisories in the PostCSS bundled inside Next 15, which only processes this repo's
  own CSS at build time. Not exploitable here, but it is unresolved until Next 16.
- Incident reports and the assistant are not in the UI (neither is wired into `core` yet).
