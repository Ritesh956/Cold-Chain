# dashboard: live zones, alerts, drift

`dashboard/`. Next.js (App Router) + React, no UI or chart library. It shows every freezer zone live,
the alert list and the drift monitor's verdicts. **All data is SYNTHETIC.**

```
browser ──► Next.js server ──/api/core/*────► core     (REST history + SSE /v1/stream)
                          └─/api/monitor/*──► monitor  (/v1/drift, polled every 30 s)
```

The browser only talks to the Next.js server, which proxies to the services (in `middleware.ts`), so neither needs
CORS. The proxy targets and keys are read **at runtime**: `CORE_URL`, `MONITOR_URL`, `CORE_API_KEY`,
`MONITOR_API_KEY`. The service keys are added on the server, so they never reach the browser. Set
`DASHBOARD_PASSWORD` to put the whole dashboard behind HTTP Basic auth (any user name).

## What it shows

- **Zone cards:** latest temperature and compressor state, a temperature chart with the -18 C limit and
  shaded spans for alert episodes, and the open alerts with the model's plain-English summary.
  Health is written out as well as coloured (Normal / Warning / Critical / No data).
- **Alerts:** newest first, with open/closed state and an **Acknowledge** button (records that someone has seen
  an alert; it stays active until the model stops alarming).
- **Incident reports:** a form that sends free text (and an optional zone) to `core`, which classifies it with the
  ml-api model and stores it; rows arrive over the live stream, with the label, the score (a ranking signal, not
  a probability) and a "needs review" flag for low scores.
- **Model drift:** PSI per model score and per (zone, feature), with the monitor's state, including
  "not enough data" and "building baseline" instead of a false "stable".
- **Connection state** (Live / Reconnecting / Offline). The page reseeds from REST after every reconnect,
  so a dropped stream leaves no gap in the charts.

## Design decisions

- **Stream first, then history.** The SSE connection opens before the history loads; duplicates are
  dropped because only strictly newer readings extend a series.
- **Events are batched** (500 ms) before rendering. At the simulator's 60x replay four zones produce
  about 240 events per second; rendering each one would saturate the page.
- **State logic is a pure module** (`lib/state.ts`) with unit tests (21, vitest): dedupe, ordering,
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
  raised by the real models), the drift table, no console errors, no horizontal scroll at 375 px. The incident form was also driven by hand
  (a safety report typed in, classified by the real int8 model, listed live).
- **Also verified:** the Docker-built image against the Compose stack ([docker-run.md](docker-run.md)): live
  zones, a critical zone with both alert texts and the fault region on the chart.
- **Not verified:** the incident panel against the Compose stack (the incident model was not registered there);
  browsers other than the in-app one; screen readers (it uses landmarks, `role=status`
  and text labels, but was not tested with one).

## Limitations

- One shared password and shared service keys, no user accounts, and Basic auth over plain HTTP exposes the
  password: put TLS in front of anything beyond localhost. The password check and key forwarding were verified
  with HTTP requests against real processes; the browser's password prompt itself was not exercised (the in-app
  browser cannot answer it).
- Keeps 24 h of history per zone (1,440 readings) with a 1 h / 6 h / 24 h chart range; no date picker, no zoom,
  nothing older than a day.
- It can acknowledge alerts but not close them or assign them.
- `npm audit` reports advisories in the PostCSS bundled inside Next 15, which only processes this repo's
  own CSS at build time. Not exploitable here, but it is unresolved until Next 16.
- The incident panel submits and lists reports, but a `safety` report does not alert anyone. The assistant is
  not in the UI (it does not exist yet).
