from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    mlflow_tracking_uri: str = "file:./mlruns"
    anomaly_model_uri: str = "models:/coldchain-anomaly@champion"
    breach_model_uri: str = "models:/coldchain-breach@champion"
    # optional: if it cannot be loaded the incident endpoint answers 503 and the rest keeps working
    incident_model_uri: str = "models:/coldchain-incident@champion"

    @classmethod
    def from_env(cls) -> Settings:
        d = cls()
        return cls(
            mlflow_tracking_uri=os.getenv("MLFLOW_TRACKING_URI", d.mlflow_tracking_uri),
            anomaly_model_uri=os.getenv("ANOMALY_MODEL_URI", d.anomaly_model_uri),
            breach_model_uri=os.getenv("BREACH_MODEL_URI", d.breach_model_uri),
            incident_model_uri=os.getenv("INCIDENT_MODEL_URI", d.incident_model_uri),
        )
