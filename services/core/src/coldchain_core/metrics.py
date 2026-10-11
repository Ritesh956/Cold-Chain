"""Prometheus metrics. One registry per app instance so tests can build many apps."""

from __future__ import annotations

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram

LATENCY_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10)


class Metrics:
    def __init__(self) -> None:
        self.registry = r = CollectorRegistry()
        self.requests = Counter(
            "http_requests_total", "HTTP requests", ["method", "route", "status"], registry=r
        )
        self.request_seconds = Histogram(
            "http_request_duration_seconds", "HTTP latency", ["route"], buckets=LATENCY_BUCKETS, registry=r
        )
        self.ingested = Counter(
            "core_readings_ingested_total", "Readings consumed, by result", ["result"], registry=r
        )
        self.poison = Counter(
            "core_ingest_poison_messages_total", "Undecodable messages dropped (terminated)", registry=r
        )
        self.ingest_errors = Counter(
            "core_ingest_errors_total", "Ingest batch failures", ["stage"], registry=r
        )
        self.scored = Counter(
            "core_scoring_total", "Scoring attempts by model and outcome", ["model", "outcome"], registry=r
        )
        self.scoring_seconds = Histogram(
            "core_scoring_seconds", "ml-api round trip", ["model"], buckets=LATENCY_BUCKETS, registry=r
        )
        self.alerts = Counter(
            "core_alert_events_total", "Alert opens and closes", ["kind", "action"], registry=r
        )
        self.active_alerts = Gauge("core_active_alerts", "Currently open alerts", ["kind"], registry=r)
        self.sse_subscribers = Gauge("core_sse_subscribers", "Connected SSE clients", registry=r)
        self.last_ingest_unixtime = Gauge(
            "core_last_ingest_unixtime", "Wall-clock time of the last stored batch", registry=r
        )
