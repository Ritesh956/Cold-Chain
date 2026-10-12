"""Model wrappers. They reuse the TRAINING code (features, detectors, debounce) so serving cannot
drift from what was evaluated; ml/tests/test_serving_parity.py guards the window contract."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from coldchain_ml.anomaly.detectors import RobustZScore
from coldchain_ml.anomaly.events_eval import apply_min_run
from coldchain_ml.anomaly.features import build_features
from coldchain_ml.breach.data import BREACH_C, RECENT_MIN, build_breach_frame

OFF_LABEL = "compressor-off duration"
SIGNAL_LABELS = {"temp": "temperature", "cur_on_360": "compressor current", "off_run": OFF_LABEL}


@dataclass(frozen=True)
class AnomalyResult:
    ts: pd.DatetimeIndex
    score: np.ndarray
    alarm: np.ndarray
    drivers: dict[str, np.ndarray]


@dataclass(frozen=True)
class BreachResult:
    ts: pd.DatetimeIndex
    state: list[str]
    at_risk: np.ndarray
    risk: np.ndarray
    alarm: np.ndarray


class AnomalyModel:
    name = "coldchain-anomaly"

    def __init__(self, config: dict[str, Any], version: str) -> None:
        self.config, self.version = config, version
        self.threshold: float = config["threshold"]
        self.min_run: int = config["min_run"]
        self.min_history: int = config["min_history_min"]
        self.detector = RobustZScore(tuple(config["signals"]))
        self.detector.stats = {k: tuple(v) for k, v in config["stats"].items()}

    @classmethod
    def from_dir(cls, path: Path, version: str) -> AnomalyModel:
        return cls(json.loads((path / "detector.json").read_text(encoding="utf-8")), version)

    def score(self, readings: pd.DataFrame) -> AnomalyResult:
        feats = build_features(readings)
        score = self.detector.score(feats)
        alarm = apply_min_run(np.nan_to_num(score, nan=-np.inf) > self.threshold, self.min_run)
        drivers = {
            s: ((feats[s] - med) / mad).to_numpy() for s, (med, mad) in self.detector.stats.items()
        }
        return AnomalyResult(pd.DatetimeIndex(readings["ts"]), score, alarm, drivers)


class BreachModel:
    name = "coldchain-breach"

    def __init__(self, model: Any, config: dict[str, Any], version: str) -> None:
        self.model, self.config, self.version = model, config, version
        self.threshold: float = config["threshold"]
        self.min_run: int = config["min_run"]
        self.min_history: int = config["min_history_min"]
        self.features: list[str] = config["features"]
        self.horizon: int = config["horizon_min"]
        self.breach_c: float = config["breach_c"]

    @classmethod
    def from_dir(cls, path: Path, version: str) -> BreachModel:
        import mlflow.lightgbm

        config = json.loads((path / "serving_config.json").read_text(encoding="utf-8"))
        return cls(mlflow.lightgbm.load_model(str(path)), config, version)

    def predict(self, readings: pd.DataFrame) -> BreachResult:
        frame = build_breach_frame(readings, build_features(readings))
        risk = self.model.predict_proba(frame[self.features])[:, 1]
        at_risk = frame["at_risk"].to_numpy(dtype=bool)
        alarm = apply_min_run((risk > self.threshold) & at_risk, self.min_run)

        temp = readings["temp_c"]
        breached = (temp > BREACH_C).fillna(False)
        recent = breached.rolling(RECENT_MIN, min_periods=1).max().astype(bool)
        state = [
            "no_reading" if pd.isna(t) else "breached" if b else "recovering" if r else "normal"
            for t, b, r in zip(temp, breached, recent, strict=True)
        ]
        return BreachResult(pd.DatetimeIndex(readings["ts"]), state, at_risk, risk, alarm)
