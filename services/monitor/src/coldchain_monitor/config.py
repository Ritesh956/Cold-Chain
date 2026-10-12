from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    core_url: str = "http://127.0.0.1:8000"
    ml_api_url: str = "http://127.0.0.1:8081"
    api_key: str = ""  # protects /v1/*; empty disables auth (development only)
    core_api_key: str = ""  # sent to core
    ml_api_key: str = ""  # sent to the ml-api
    window_min: int = 1440  # minutes of readings compared with the reference
    min_samples: int = 200  # fewer readings than this: "insufficient_data", never a verdict
    interval_s: float = 30.0
    warn_psi: float = 0.10
    drift_psi: float = 0.25
    # prediction drift: the first `baseline_count` scored minutes are the baseline, later ones
    # (the last `score_window_s` of wall clock) are compared against it
    baseline_count: int = 500
    score_window_s: float = 3600.0
    score_min_count: int = 100

    @classmethod
    def from_env(cls) -> Settings:
        d = cls()

        def f(name: str, default, cast=str):
            raw = os.getenv(name)
            return default if raw is None else cast(raw)

        return cls(
            core_url=f("CORE_URL", d.core_url),
            ml_api_url=f("ML_API_URL", d.ml_api_url),
            api_key=f("API_KEY", d.api_key),
            core_api_key=f("CORE_API_KEY", d.core_api_key),
            ml_api_key=f("ML_API_KEY", d.ml_api_key),
            window_min=f("DRIFT_WINDOW_MIN", d.window_min, int),
            min_samples=f("DRIFT_MIN_SAMPLES", d.min_samples, int),
            interval_s=f("DRIFT_INTERVAL_S", d.interval_s, float),
            warn_psi=f("DRIFT_WARN_PSI", d.warn_psi, float),
            drift_psi=f("DRIFT_PSI", d.drift_psi, float),
            baseline_count=f("SCORE_BASELINE_COUNT", d.baseline_count, int),
            score_window_s=f("SCORE_WINDOW_S", d.score_window_s, float),
            score_min_count=f("SCORE_MIN_COUNT", d.score_min_count, int),
        )
