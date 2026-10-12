"""Anomaly scorers. Each returns a score per row (higher = more anomalous, NaN = no opinion).

Thresholds are NOT chosen here; they are tuned on a validation history in `run.py`.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import IsolationForest

from coldchain_ml.anomaly.features import MODEL_FEATURES


class Detector(Protocol):
    name: str
    needs_labels: bool

    def fit(self, feats: pd.DataFrame, labels: pd.Series | None = None) -> None: ...
    def score(self, feats: pd.DataFrame) -> np.ndarray: ...


class GlobalZScore:
    """Baseline 1: |temp - mean| / std using statistics from the training history."""

    name = "z-score (baseline)"
    needs_labels = False

    def fit(self, feats: pd.DataFrame, labels: pd.Series | None = None) -> None:
        t = feats["temp"].dropna()
        self.mu, self.sigma = float(t.mean()), float(t.std())

    def score(self, feats: pd.DataFrame) -> np.ndarray:
        return (feats["temp"] - self.mu).abs().to_numpy() / self.sigma


OFF_Z = 5.0  # z-equivalent of a compressor-off run exactly as long as the reference limit
OFF_QUANTILE = 0.995
OFF_MARGIN = 1.25


class RobustZScore:
    """Max of robust z-scores (median / 1.4826*MAD) over the chosen signals.

    `off_run` is the compressor-off rule: minutes the compressor has been continuously off. Normal
    thermostat and defrost cycles end within a known time, so a run far beyond that is a power cut
    even while the temperature is still inside its band. It is expressed as a z-equivalent
    (OFF_Z * run / limit, median 0) so it shares the threshold and the driver formula with the other
    signals. The limit is OFF_MARGIN x the 99.5th percentile of the run length over all training
    minutes, so it survives training data in which power cuts are under 0.5% of minutes.

    Median/MAD instead of mean/std because the training history itself contains
    anomalies (e.g. days of compressor drift) that would inflate a plain std.
    `cur_on_360` is the mean compressor current while running over the last 6 h:
    an equipment-health signal that moves long before the freezer temperature does."""

    needs_labels = False

    def __init__(self, signals: tuple[str, ...]) -> None:
        self.signals = signals
        self.name = "robust z-score (" + " + ".join(
            {"temp": "temp", "cur_on_360": "compressor current", "off_run": "compressor-off duration"}[s]
            for s in signals
        ) + ")"

    def fit(self, feats: pd.DataFrame, labels: pd.Series | None = None) -> None:
        self.stats = {}
        for s in self.signals:
            x = feats[s].dropna()
            if s == "off_run":
                limit = OFF_MARGIN * float(x.quantile(OFF_QUANTILE))
                self.stats[s] = (0.0, limit / OFF_Z)
                continue
            med = float(x.median())
            self.stats[s] = (med, 1.4826 * float((x - med).abs().median()))

    def score(self, feats: pd.DataFrame) -> np.ndarray:
        zs = [
            ((feats[s] - med).abs() / mad).to_numpy() for s, (med, mad) in self.stats.items()
        ]
        return np.nanmax(np.vstack(zs), axis=0)


class RollingZScore:
    """Baseline 2: z-score against the trailing 6 h. Adapts to slow change, so a drift
    that creeps over days looks 'normal' to it."""

    name = "rolling z-score (baseline)"
    needs_labels = False

    def fit(self, feats: pd.DataFrame, labels: pd.Series | None = None) -> None:
        return None

    def score(self, feats: pd.DataFrame) -> np.ndarray:
        std = feats["temp_rstd_360"].clip(lower=0.3)
        return ((feats["temp"] - feats["temp_rmean_360"]).abs() / std).to_numpy()


class IsolationForestDetector:
    """Unsupervised: learns what 'normal' windows look like; no labels needed."""

    name = "isolation forest"
    needs_labels = False

    def __init__(
        self, seed: int = 0, features: list[str] | None = None, stride: int = 3
    ) -> None:
        self.features = features or MODEL_FEATURES
        self.stride = stride
        self.model = IsolationForest(n_estimators=200, max_samples=4096, random_state=seed, n_jobs=-1)

    def fit(self, feats: pd.DataFrame, labels: pd.Series | None = None) -> None:
        x = feats[self.features].dropna().iloc[:: self.stride]
        self.model.fit(x)

    def score(self, feats: pd.DataFrame) -> np.ndarray:
        x = feats[self.features]
        ok = x.notna().all(axis=1).to_numpy()
        out = np.full(len(x), np.nan)
        out[ok] = -self.model.score_samples(x[ok])
        return out


class SupervisedLGBM:
    """UPPER BOUND, not a deployable choice: needs per-minute labels, which a real site rarely
    has. On synthetic data it also learns the simulator's own signatures."""

    name = "lightgbm supervised (upper bound)"
    needs_labels = True

    def __init__(self, seed: int = 0) -> None:
        self.model = LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.8,
            random_state=seed,
            n_jobs=-1,
            verbose=-1,
        )

    def fit(self, feats: pd.DataFrame, labels: pd.Series | None = None) -> None:
        assert labels is not None
        x = feats[MODEL_FEATURES]
        ok = x.notna().all(axis=1)
        self.model.fit(x[ok], labels[ok].astype(int))

    def score(self, feats: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(feats[MODEL_FEATURES])[:, 1]


def all_detectors(seed: int = 0) -> list[Detector]:
    return [
        GlobalZScore(),
        RollingZScore(),
        RobustZScore(("temp",)),
        RobustZScore(("temp", "cur_on_360")),
        RobustZScore(("temp", "cur_on_360", "off_run")),
        IsolationForestDetector(seed),
        SupervisedLGBM(seed),
    ]
