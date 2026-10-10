import numpy as np
import pandas as pd
from coldchain_ml.anomaly.detectors import RobustZScore
from coldchain_ml.anomaly.features import MODEL_FEATURES, build_features


def _readings(n=3000, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "ts": pd.date_range("2025-01-01", periods=n, freq="min", tz="UTC"),
            "zone": "A",
            "temp_c": -25 + rng.normal(0, 0.5, n),
            "compressor_on": rng.random(n) < 0.5,
            "compressor_current_a": 12 + rng.normal(0, 0.3, n),
        }
    )


def test_features_are_causal():
    """Changing the future must not change any feature in the past."""
    a = _readings()
    b = a.copy()
    b.loc[2000:, ["temp_c", "compressor_current_a"]] += 50.0
    fa, fb = build_features(a), build_features(b)
    pd.testing.assert_frame_equal(fa.iloc[:2000], fb.iloc[:2000])


def test_features_aligned_and_complete_after_warmup():
    r = _readings()
    f = build_features(r)
    assert f.index.equals(r.index)
    assert f[MODEL_FEATURES].iloc[1500:].notna().all().all()


def test_robust_z_flags_shift_and_ignores_outlier_contamination():
    r = _readings()
    f = build_features(r)
    f.loc[:200, "cur_on_360"] = 99.0  # contaminated training tail must not move the median/MAD
    det = RobustZScore(("cur_on_360",))
    det.fit(f)
    healthy = f.copy()
    healthy["cur_on_360"] = 12.0
    shifted = f.copy()
    shifted["cur_on_360"] = 16.0
    assert np.nanmin(det.score(shifted)) > np.nanmax(det.score(healthy))
