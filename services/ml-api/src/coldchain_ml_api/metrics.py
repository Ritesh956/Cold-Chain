"""Prometheus metrics. One registry per app instance so tests can build many apps."""

from __future__ import annotations

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram

LATENCY_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5)
ANOMALY_SCORE_BUCKETS = (0.5, 1, 2, 3, 4, 5, 7, 10, 20, 50, 100)  # robust z
BREACH_SCORE_BUCKETS = (0.001, 0.01, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 0.95, 1.0)


class Metrics:
    def __init__(self) -> None:
        self.registry = r = CollectorRegistry()
        self.requests = Counter(
            "http_requests_total", "HTTP requests", ["method", "route", "status"], registry=r
        )
        self.request_seconds = Histogram(
            "http_request_duration_seconds", "HTTP latency", ["route"], buckets=LATENCY_BUCKETS, registry=r
        )
        self.inference_seconds = Histogram(
            "model_inference_seconds",
            "Feature build + model time",
            ["model"],
            buckets=LATENCY_BUCKETS,
            registry=r,
        )
        self.predictions = Counter(
            "model_predictions_total", "Scored minutes returned, by outcome", ["model", "outcome"], registry=r
        )
        # Score distributions of the latest scored minute: raw material for prediction-drift checks.
        self.anomaly_score = Histogram(
            "anomaly_score",
            "Latest-minute anomaly score (robust z)",
            buckets=ANOMALY_SCORE_BUCKETS,
            registry=r,
        )
        self.breach_risk = Histogram(
            "breach_risk_score", "Latest-minute breach risk score", buckets=BREACH_SCORE_BUCKETS, registry=r
        )
        self.rejected = Counter("input_rejected_total", "Rejected inputs", ["reason"], registry=r)
        self.model_info = Gauge(
            "model_info", "Loaded model version (value is always 1)", ["model", "version"], registry=r
        )
