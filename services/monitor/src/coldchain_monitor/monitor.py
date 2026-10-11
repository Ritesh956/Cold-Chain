"""One drift check: input drift per (zone, feature), prediction drift per model score."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import asdict
from typing import Any

from coldchain_monitor.config import Settings
from coldchain_monitor.drift import Profile, bin_counts, psi_from_counts, status
from coldchain_monitor.metrics import Metrics
from coldchain_monitor.prom import histogram_buckets
from coldchain_monitor.scores import ScoreTracker
from coldchain_monitor.sources import DataSource, SourceError

log = logging.getLogger("coldchain_monitor")

# score histogram exported by the ml-api -> label used here
SCORE_HISTOGRAMS = {"anomaly_score": "anomaly", "breach_risk_score": "breach"}


def _values(readings: list[dict], feature: str) -> list[float]:
    out = []
    for r in readings:
        v = r.get(feature)
        if v is None:  # sensor dropout
            continue
        out.append(float(v))  # bools become 0.0 / 1.0
    return out


class DriftMonitor:
    def __init__(
        self,
        source: DataSource,
        profiles: dict[str, Profile],
        settings: Settings,
        metrics: Metrics,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.source, self.profiles, self.s, self.m, self.clock = source, profiles, settings, metrics, clock
        self.trackers = {
            label: ScoreTracker(
                settings.baseline_count,
                settings.score_window_s,
                settings.score_min_count,
                settings.warn_psi,
                settings.drift_psi,
            )
            for label in SCORE_HISTOGRAMS.values()
        }
        self.report: dict[str, Any] = {"checked_at": None, "inputs": [], "predictions": [], "errors": []}

    async def run_once(self) -> dict[str, Any]:
        errors: list[str] = []
        inputs = await self._inputs(errors)
        predictions = await self._predictions(errors)
        self.report = {
            "checked_at": self.clock(),
            "thresholds": {"warn": self.s.warn_psi, "drift": self.s.drift_psi},
            "window_min": self.s.window_min,
            "inputs": inputs,
            "predictions": predictions,
            "errors": errors,
        }
        for e in errors:
            self.m.check_errors.inc()
            log.warning("drift check: %s", e)
        self.m.last_check.set(self.clock())
        return self.report

    async def _inputs(self, errors: list[str]) -> list[dict]:
        rows: list[dict] = []
        try:
            zones = await self.source.zones()
        except SourceError as e:
            errors.append(str(e))
            return rows
        for zone in zones:
            try:
                readings = await self.source.readings(zone, self.s.window_min)
            except SourceError as e:
                errors.append(str(e))
                continue
            for feature, profile in self.profiles.items():
                vals = _values(readings, feature)
                n = len(vals)
                if n < self.s.min_samples:
                    row = {"zone": zone, "feature": feature, "psi": None, "n": n}
                    row["state"] = "insufficient_data"
                else:
                    value = psi_from_counts(profile.props, bin_counts(vals, profile.edges))
                    state = status(value, self.s.warn_psi, self.s.drift_psi)
                    row = {"zone": zone, "feature": feature, "psi": value, "n": n, "state": state}
                    self.m.input_psi.labels(zone, feature).set(value)
                self.m.input_drift.labels(zone, feature).set(1.0 if row["state"] == "drift" else 0.0)
                self.m.input_samples.labels(zone, feature).set(n)
                rows.append(row)
        return rows

    async def _predictions(self, errors: list[str]) -> list[dict]:
        try:
            text = await self.source.ml_metrics()
        except SourceError as e:
            errors.append(str(e))
            return []
        rows: list[dict] = []
        now = self.clock()
        for metric, label in SCORE_HISTOGRAMS.items():
            parsed = histogram_buckets(text, metric)
            if parsed is None:
                continue
            res = self.trackers[label].observe(now, *parsed)
            row = {"model": label, **asdict(res)}
            if res.psi is not None:
                self.m.score_psi.labels(label).set(res.psi)
            self.m.score_drift.labels(label).set(1.0 if res.state == "drift" else 0.0)
            rows.append(row)
        return rows
