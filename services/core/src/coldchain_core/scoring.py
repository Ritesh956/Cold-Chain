"""Score zones with the ml-api and turn the results into alerts.

Position is tracked per (zone, model) in EVENT time. The scorer scores every minute since the
last position (`return_last=k`), so a short alarm between two ticks is not missed. The position
advances only after the ml-api answered and the alerts were stored: an ml-api outage delays
alerts but never silently skips minutes.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta

from coldchain_schemas import AlertKind

from coldchain_core.alerts import AlertEngine, AlertEvent
from coldchain_core.config import Settings
from coldchain_core.events import AlertPublisher
from coldchain_core.metrics import Metrics
from coldchain_core.mlclient import MlApi, MlApiError
from coldchain_core.repo import Repo
from coldchain_core.windows import build_window

log = logging.getLogger("coldchain_core.scoring")
MINUTE = timedelta(minutes=1)
MODELS = ("anomaly", "breach")
# ml-api model name -> key used in /v1/models
MODEL_KEY = {"anomaly": "coldchain-anomaly", "breach": "coldchain-breach"}


class Scorer:
    def __init__(
        self,
        repo: Repo,
        ml: MlApi,
        engine: AlertEngine,
        publisher: AlertPublisher,
        metrics: Metrics,
        settings: Settings,
    ) -> None:
        self.repo, self.ml, self.engine = repo, ml, engine
        self.publisher, self.metrics, self.settings = publisher, metrics, settings
        self.min_history: dict[str, int] = {}

    async def start(self) -> None:
        """Learn each model's required history from the ml-api; adopt alerts open before a restart."""
        info = await self.ml.model_info()
        self.min_history = {m: int(info[MODEL_KEY[m]]["min_history_min"]) for m in MODELS}
        self.engine.restore(await self.repo.open_alerts())
        self._refresh_gauge()

    async def tick(self) -> None:
        if not self.min_history:
            await self.start()
        for zone in await self.repo.zones():
            for model in MODELS:
                await self.score_zone(zone, model)

    async def score_zone(self, zone: str, model: str) -> None:
        latest = await self.repo.latest(zone)
        if latest is None:
            return
        last = await self.repo.get_scoring_state(zone, model)
        if last is None:  # first sight of this zone: score only the newest minute, no backfill
            end, k = latest.ts, 1
        else:
            behind = int((latest.ts - last) / MINUTE)
            if behind < self.settings.score_every_min:
                return
            k = min(behind, self.settings.max_chunk_min)
            end = last + k * MINUTE

        minutes = self.min_history[model] + k - 1
        rows = await self.repo.window(zone, end, minutes)
        window = build_window(rows, end, minutes, self.settings.min_coverage)
        if window is None:
            # Not enough stored history (warm-up, or an outage inside the window). The minutes are
            # unscorable, so move on instead of retrying the same window forever.
            self.metrics.scored.labels(model, "skipped_insufficient_data").inc()
            await self.repo.set_scoring_state(zone, model, end)
            return

        call = self.ml.anomaly if model == "anomaly" else self.ml.breach
        t0 = time.perf_counter()
        try:
            resp = await call(zone, window, k)
        except MlApiError as e:
            self.metrics.scored.labels(model, "ml_error_retry" if e.retryable else "ml_error_rejected").inc()
            log.warning("ml-api %s scoring failed for zone %s: %s", model, zone, e)
            return  # position not advanced
        self.metrics.scoring_seconds.labels(model).observe(time.perf_counter() - t0)

        version = resp["model"]["version"]
        for p in resp["points"]:
            await self._apply(zone, model, p, version)
        await self.repo.set_scoring_state(zone, model, end)
        self.metrics.scored.labels(model, "ok").inc()
        self._refresh_gauge()

    async def _apply(self, zone: str, model: str, p: dict, version: str) -> None:
        ts = datetime.fromisoformat(p["ts"])
        if model == "anomaly":
            events = self.engine.observe(
                zone, AlertKind.ANOMALY, ts, p["alarm"], p["score"], p["summary"], version
            )
        else:
            events = self.engine.observe(
                zone, AlertKind.BREACH_RISK, ts, p["alarm"], p["risk_score"], p["summary"], version
            )
            breached = p["state"] == "breached"
            events += self.engine.observe(
                zone, AlertKind.BREACHED, ts, breached, None, p["summary"] if breached else None, version
            )
        for ev in events:
            await self._emit(ev)

    async def _emit(self, ev: AlertEvent) -> None:
        saved = await self.repo.save_alert(ev.alert)
        if ev.action == "opened":
            self.engine.assign_id(saved.zone, saved.kind, saved.id)
        self.metrics.alerts.labels(saved.kind.value, ev.action).inc()
        await self.publisher.publish(AlertEvent(ev.action, saved))

    def _refresh_gauge(self) -> None:
        counts = {k.value: 0 for k in AlertKind}
        for a in self.engine.active:
            counts[a.kind.value] += 1
        for kind, n in counts.items():
            self.metrics.active_alerts.labels(kind).set(n)
