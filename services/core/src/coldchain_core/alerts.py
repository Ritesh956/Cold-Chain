"""Alert lifecycle: turn per-minute alarm flags into alert episodes with hysteresis.

A flapping model must not spam operators, so an alert opens on the first alarm minute and closes
only after `close_after_min` consecutive quiet minutes. Time is EVENT time (reading timestamps).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Literal

from coldchain_schemas import Alert, AlertKind

SEVERITY = {
    AlertKind.BREACHED: "critical",
    AlertKind.BREACH_RISK: "warning",
    AlertKind.ANOMALY: "warning",
}


@dataclass(frozen=True)
class AlertEvent:
    action: Literal["opened", "closed"]
    alert: Alert


@dataclass
class _Open:
    alert: Alert
    last_alarm_ts: datetime
    quiet_minutes: int = 0


class AlertEngine:
    def __init__(self, close_after_min: int = 5) -> None:
        self.close_after_min = close_after_min
        self._open: dict[tuple[str, AlertKind], _Open] = {}

    def restore(self, active: list[Alert]) -> None:
        """Re-adopt alerts that were open before a restart."""
        for a in active:
            self._open[(a.zone, a.kind)] = _Open(a, last_alarm_ts=a.opened_at)

    def assign_id(self, zone: str, kind: AlertKind, alert_id: int) -> None:
        """Record the database id of a freshly opened alert so its close updates the same row."""
        cur = self._open.get((zone, kind))
        if cur is not None:
            cur.alert = cur.alert.model_copy(update={"id": alert_id})

    @property
    def active(self) -> list[Alert]:
        return [o.alert for o in self._open.values()]

    def observe(
        self,
        zone: str,
        kind: AlertKind,
        ts: datetime,
        alarm: bool,
        score: float | None = None,
        summary: str | None = None,
        model_version: str | None = None,
    ) -> list[AlertEvent]:
        """Feed one minute for one (zone, kind). Minutes must arrive in time order."""
        key = (zone, kind)
        cur = self._open.get(key)

        if alarm:
            if cur is None:
                alert = Alert(
                    zone=zone,
                    kind=kind,
                    severity=SEVERITY[kind],
                    opened_at=ts,
                    peak_score=score,
                    summary=summary,
                    model_version=model_version,
                )
                self._open[key] = _Open(alert, last_alarm_ts=ts)
                return [AlertEvent("opened", alert)]
            cur.last_alarm_ts, cur.quiet_minutes = ts, 0
            if score is not None and (cur.alert.peak_score is None or score > cur.alert.peak_score):
                cur.alert = cur.alert.model_copy(update={"peak_score": score})
            return []

        if cur is None:
            return []
        cur.quiet_minutes += 1
        if cur.quiet_minutes >= self.close_after_min:
            closed = cur.alert.model_copy(update={"closed_at": cur.last_alarm_ts + timedelta(minutes=1)})
            del self._open[key]
            return [AlertEvent("closed", closed)]
        return []
