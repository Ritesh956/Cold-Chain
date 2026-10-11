from dataclasses import replace

import pandas as pd
import pytest
from coldchain_ml.anomaly.detectors import RobustZScore
from coldchain_ml.anomaly.features import build_features
from coldchain_ml.breach.data import BREACH_C, BREACH_FEATURES, build_breach_frame
from coldchain_ml_api.app import create_app
from coldchain_ml_api.models import AnomalyModel, BreachModel
from coldchain_ml_api.store import Models
from coldchain_simulator import SimConfig, generate
from fastapi.testclient import TestClient
from lightgbm import LGBMClassifier

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


def anomaly_config(sim: Sim, threshold: float = 4.0) -> dict:
    det = RobustZScore(("temp", "cur_on_360"))
    det.fit(sim.feats)
    return {
        "type": "robust_z_score",
        "signals": ["temp", "cur_on_360"],
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


@pytest.fixture(scope="session")
def sim() -> Sim:
    return Sim()


@pytest.fixture(scope="session")
def models(sim) -> Models:
    clf, bconf = breach_model_and_config(sim)
    return Models(AnomalyModel(anomaly_config(sim), "test"), BreachModel(clf, bconf, "test"))


@pytest.fixture()
def client(models):
    with TestClient(create_app(models=models)) as c:
        yield c
