"""Prometheus metrics. One registry per app instance so tests can build many apps."""

from __future__ import annotations

from prometheus_client import CollectorRegistry, Counter, Gauge

ZF = ["zone", "feature"]


class Metrics:
    def __init__(self) -> None:
        self.registry = r = CollectorRegistry()
        self.input_psi = Gauge("drift_input_psi", "PSI of a live input vs the reference", ZF, registry=r)
        self.input_drift = Gauge(
            "drift_input_flag", "1 when the input PSI is at or above the drift threshold", ZF, registry=r
        )
        self.input_samples = Gauge("drift_input_samples", "Readings behind the latest PSI", ZF, registry=r)
        self.score_psi = Gauge(
            "drift_score_psi", "PSI of recent model scores vs the baseline", ["model"], registry=r
        )
        self.score_drift = Gauge(
            "drift_score_flag", "1 when the score PSI is at the drift threshold", ["model"], registry=r
        )
        self.last_check = Gauge("drift_last_check_unixtime", "Wall-clock time of the last check", registry=r)
        self.check_errors = Counter("drift_check_errors_total", "Source failures during checks", registry=r)
