"""Drift statistics. Pure functions, no I/O.

PSI (population stability index) compares two distributions over the same bins:
sum((cur - ref) * ln(cur / ref)). The usual reading is < 0.1 stable, 0.1 to 0.25 moderate shift,
>= 0.25 large shift. Those cut-offs are a convention, not a law; they are settings here.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

EPS = 1e-4  # floor for an empty bin, so ln() stays finite
WARN_PSI = 0.10
DRIFT_PSI = 0.25


@dataclass(frozen=True)
class Profile:
    """Reference distribution of one feature: interior bin edges and the share of data per bin."""

    feature: str
    edges: tuple[float, ...]  # n_bins - 1 interior edges, ascending
    props: tuple[float, ...]  # n_bins shares summing to 1
    n: int  # reference sample size

    def to_dict(self) -> dict:
        return {"feature": self.feature, "edges": list(self.edges), "props": list(self.props), "n": self.n}

    @classmethod
    def from_dict(cls, d: dict) -> Profile:
        p = cls(d["feature"], tuple(d["edges"]), tuple(d["props"]), int(d["n"]))
        if len(p.props) != len(p.edges) + 1 or abs(sum(p.props) - 1) > 1e-6:
            raise ValueError(f"inconsistent profile for {p.feature!r}")
        return p


def make_profile(feature: str, values: Sequence[float], n_bins: int = 10) -> Profile:
    """Quantile bins from the reference, so every reference bin holds about the same share.
    Repeated quantiles (a discrete or saturated feature) collapse into fewer bins."""
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        raise ValueError(f"no finite values for {feature!r}")
    edges = np.unique(np.quantile(x, np.linspace(0, 1, n_bins + 1)[1:-1]))
    props = bin_counts(x, edges) / x.size
    return Profile(feature, tuple(float(e) for e in edges), tuple(float(p) for p in props), int(x.size))


def bin_counts(values: Sequence[float], edges: Sequence[float]) -> np.ndarray:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    idx = np.searchsorted(np.asarray(edges, dtype=float), x, side="right")
    return np.bincount(idx, minlength=len(edges) + 1).astype(float)


def psi_from_counts(ref_props: Sequence[float], cur_counts: Sequence[float]) -> float:
    cur = np.asarray(cur_counts, dtype=float)
    ref = np.asarray(ref_props, dtype=float)
    if cur.shape != ref.shape:
        raise ValueError("reference and current must use the same bins")
    total = cur.sum()
    if total <= 0:
        raise ValueError("no current observations")
    r = np.maximum(ref / ref.sum(), EPS)
    c = np.maximum(cur / total, EPS)
    return float(np.sum((c - r) * np.log(c / r)))


def psi(profile: Profile, values: Sequence[float]) -> float:
    return psi_from_counts(profile.props, bin_counts(values, profile.edges))


def status(value: float, warn: float = WARN_PSI, drift: float = DRIFT_PSI) -> str:
    if not math.isfinite(value):
        return "unknown"
    if value >= drift:
        return "drift"
    if value >= warn:
        return "warn"
    return "ok"
