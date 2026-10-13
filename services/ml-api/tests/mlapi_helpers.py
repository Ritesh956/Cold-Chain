"""Shared test helpers for the ml-api tests."""

from dataclasses import replace

import numpy as np
import pandas as pd
from coldchain_ml.anomaly.detectors import RobustZScore
from coldchain_ml.anomaly.features import build_features
from coldchain_ml.breach.data import BREACH_C, BREACH_FEATURES, build_breach_frame
from coldchain_ml.forecast.features import SERVED_FEATURES, history_frame
from coldchain_simulator import SimConfig, generate
from lightgbm import LGBMClassifier, LGBMRegressor

CFG = replace(
    SimConfig(), days=20, zones=("A",), seed=3, n_door_left_open=3, n_power_cut=1, n_compressor_drift=0
)


class Sim:
    """A short simulated history with ground truth, plus helpers to cut request windows."""

    def __init__(self) -> None:
        self.readings, self.events = generate(CFG)
        self.feats = build_features(self.readings)
        self.frame = build_breach_frame(self.readings, self.feats)

    def index_at(self, ts: pd.Timestamp) -> int:
        return int(self.readings["ts"].searchsorted(ts))

    def payload(self, end: int, n: int = 1500, return_last: int = 1, zone: str = "A") -> dict:
        """Request body for the n readings ending at row `end` (inclusive)."""
        w = self.readings.iloc[end - n + 1 : end + 1]
        return {
            "zone": zone,
            "return_last": return_last,
            "readings": [
                {
                    "ts": r.ts.isoformat(),
                    "temp_c": None if pd.isna(r.temp_c) else float(r.temp_c),
                    "compressor_on": bool(r.compressor_on),
                    "compressor_current_a": float(r.compressor_current_a),
                }
                for r in w.itertuples()
            ],
        }

    def event(self, kind: str, i: int = 0) -> pd.Series:
        return self.events[self.events["type"] == kind].sort_values("start").iloc[i]


def anomaly_config(
    sim: Sim, threshold: float = 4.0, signals: tuple[str, ...] = ("temp", "cur_on_360")
) -> dict:
    det = RobustZScore(signals)
    det.fit(sim.feats)
    return {
        "type": "robust_z_score",
        "signals": list(signals),
        "stats": {k: list(v) for k, v in det.stats.items()},
        "threshold": threshold,
        "min_run": 5,
        "min_history_min": 400,
        "data": "SYNTHETIC",
    }


def breach_model_and_config(sim: Sim) -> tuple[LGBMClassifier, dict]:
    ok = sim.frame["at_risk"] & sim.frame["y"].notna() & sim.frame[BREACH_FEATURES].notna().all(axis=1)
    clf = LGBMClassifier(n_estimators=60, num_leaves=15, min_child_samples=50, random_state=0, verbose=-1)
    clf.fit(sim.frame.loc[ok, BREACH_FEATURES], sim.frame.loc[ok, "y"].astype(int))
    config = {
        "type": "lightgbm_breach",
        "features": BREACH_FEATURES,
        "threshold": 0.3,
        "min_run": 3,
        "horizon_min": 30,
        "breach_c": BREACH_C,
        "min_history_min": 1500,
        "data": "SYNTHETIC",
    }
    return clf, config


class _Enc:
    def __init__(self, ids: list[int], mask: list[int]) -> None:
        self.ids, self.attention_mask = ids, mask


class StubTokenizer:
    """word -> id by a small vocabulary; pads to the longest text like the real tokenizer."""

    VOCAB = {
        "crushed": 1,
        "torn": 1,
        "temperature": 2,
        "alarm": 2,
        "late": 3,
        "delayed": 3,
        "injury": 4,
        "slipped": 4,
    }

    def encode_batch(self, texts: list[str]) -> list[_Enc]:
        rows = [[self.VOCAB.get(w, 9) for w in t.lower().split()] or [9] for t in texts]
        width = max(len(r) for r in rows)
        return [_Enc(r + [0] * (width - len(r)), [1] * len(r) + [0] * (width - len(r))) for r in rows]


class StubSession:
    """logits favour the class whose keyword ids appear (1 damage, 2 temperature, 3 delay, 4 safety)."""

    def run(self, names, feed):
        import numpy as np

        ids = feed["input_ids"]
        logits = np.zeros((ids.shape[0], 4))
        for i, row in enumerate(ids):
            for tok in row:
                if 1 <= tok <= 4:
                    logits[i, tok - 1] += 3.0
        return [logits]


def make_incident_model(review_below: float = 0.7):
    from coldchain_ml.nlp.serving import IncidentModel

    config = {
        "labels": ["damage", "temperature", "delay", "safety"],
        "review_below": review_below,
        "max_chars": 200,
    }
    return IncidentModel(StubSession(), StubTokenizer(), config, "test")


def make_forecast_model(seed: int = 0):
    """A small LightGBM trained through the same feature path as serving; returns (model wrapper, units)."""
    from coldchain_ml_api.models import ForecastModel

    rng = np.random.default_rng(seed)
    n_items, n_days = 30, 300
    week = np.array([1.0, 0.9, 0.9, 0.9, 1.0, 1.2, 1.3])
    base = rng.uniform(3, 30, size=(n_items, 1))
    units = rng.poisson(base * np.tile(week, n_days // 7 + 1)[:n_days]).astype(float)
    start = pd.Timestamp("2024-01-01")
    frames, targets = [], []
    for o in range(100, n_days - 28, 10):
        frames.append(history_frame(units[:, : o + 1], start + pd.Timedelta(days=o)))
        targets.append(units[:, o + 1 : o + 29].ravel())
    model = LGBMRegressor(n_estimators=60, num_leaves=15, min_child_samples=20, random_state=0, verbose=-1)
    model.fit(pd.concat(frames), np.concatenate(targets))
    config = {
        "type": "lightgbm_outflow_forecast",
        "features": SERVED_FEATURES,
        "horizon_days": 28,
        "min_history_days": 90,
        "max_history_days": 400,
        "data": "SYNTHETIC (test)",
        "reported_test": {},
    }
    return ForecastModel(model, config, "test"), units, start
