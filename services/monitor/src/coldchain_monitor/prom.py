"""Read histograms out of another service's Prometheus text exposition (the ml-api's /metrics)."""

from __future__ import annotations

import math

from prometheus_client.parser import text_string_to_metric_families


def histogram_buckets(text: str, name: str) -> tuple[list[float], list[float]] | None:
    """Return (upper bounds, cumulative counts) of an unlabelled histogram, or None if absent.
    The +Inf bucket is last."""
    for family in text_string_to_metric_families(text):
        if family.name != name:
            continue
        pairs = sorted(
            (float(s.labels["le"]), float(s.value)) for s in family.samples if s.name == f"{name}_bucket"
        )
        if not pairs:
            return None
        return [p[0] for p in pairs], [p[1] for p in pairs]
    return None


def per_bucket(cumulative: list[float]) -> list[float]:
    """Cumulative -> per-bucket counts."""
    return [c - (cumulative[i - 1] if i else 0.0) for i, c in enumerate(cumulative)]


def finite_bounds(bounds: list[float]) -> list[float]:
    return [b for b in bounds if math.isfinite(b)]
