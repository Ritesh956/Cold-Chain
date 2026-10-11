from __future__ import annotations

import math

import numpy as np
import pytest
from coldchain_monitor.drift import make_profile, psi, psi_from_counts, status
from coldchain_monitor.prom import histogram_buckets, per_bucket

RNG = np.random.default_rng(0)


def test_same_distribution_has_tiny_psi():
    ref = RNG.normal(-25, 1, 20_000)
    p = make_profile("t", ref)
    assert psi(p, RNG.normal(-25, 1, 5_000)) < 0.02


def test_shifted_distribution_is_flagged():
    p = make_profile("t", RNG.normal(-25, 1, 20_000))
    shifted = psi(p, RNG.normal(-23.5, 1, 5_000))
    assert shifted > 0.25
    assert status(shifted) == "drift"


def test_psi_is_zero_for_identical_proportions():
    assert psi_from_counts([0.25] * 4, [10, 10, 10, 10]) == pytest.approx(0.0)


def test_empty_bin_does_not_blow_up():
    value = psi_from_counts([0.5, 0.5], [100, 0])
    assert math.isfinite(value) and value > 0.25


def test_profile_collapses_repeated_quantiles():
    # a saturated/discrete feature: most values identical
    p = make_profile("c", [0.0] * 900 + list(np.linspace(1, 10, 100)))
    assert len(p.props) == len(p.edges) + 1
    assert sum(p.props) == pytest.approx(1.0)


def test_boolean_feature_uses_two_bins():
    p = make_profile("on", [1.0] * 60 + [0.0] * 40, n_bins=2)
    assert p.props == (pytest.approx(0.4), pytest.approx(0.6))
    assert psi(p, [1.0] * 6 + [0.0] * 4) == pytest.approx(0.0, abs=1e-9)
    assert psi(p, [1.0] * 10) > 0.25  # a compressor that never switches off


def test_nan_values_are_ignored():
    p = make_profile("t", [1.0, 2.0, float("nan"), 3.0, 4.0], n_bins=2)
    assert p.n == 4


def test_profile_roundtrip_and_validation():
    p = make_profile("t", RNG.normal(size=500))
    assert type(p).from_dict(p.to_dict()) == p
    bad = p.to_dict()
    bad["props"] = bad["props"][:-1]
    with pytest.raises(ValueError):
        type(p).from_dict(bad)


def test_status_boundaries():
    assert status(0.099) == "ok"
    assert status(0.10) == "warn"
    assert status(0.249) == "warn"
    assert status(0.25) == "drift"
    assert status(float("nan")) == "unknown"


def test_mismatched_bins_rejected():
    with pytest.raises(ValueError):
        psi_from_counts([0.5, 0.5], [1, 2, 3])


METRICS_TEXT = """\
# HELP anomaly_score Latest-minute anomaly score
# TYPE anomaly_score histogram
anomaly_score_bucket{le="1.0"} 5.0
anomaly_score_bucket{le="2.0"} 8.0
anomaly_score_bucket{le="+Inf"} 10.0
anomaly_score_count 10.0
anomaly_score_sum 12.0
"""


def test_histogram_parsing():
    bounds, cum = histogram_buckets(METRICS_TEXT, "anomaly_score")
    assert bounds == [1.0, 2.0, math.inf]
    assert per_bucket(cum) == [5.0, 3.0, 2.0]
    assert histogram_buckets(METRICS_TEXT, "missing") is None
