"""Load models from the MLflow registry by URI (models:/<name>@<alias>)."""

from __future__ import annotations

import re
import tempfile
from pathlib import Path

import mlflow
from mlflow import MlflowClient

from coldchain_ml_api.config import Settings
from coldchain_ml_api.models import AnomalyModel, BreachModel

_URI = re.compile(r"^models:/(?P<name>[^@/]+)@(?P<alias>[^@/]+)$")


def _fetch(uri: str) -> tuple[Path, str]:
    m = _URI.match(uri)
    if not m:
        raise ValueError(f"expected models:/<name>@<alias>, got {uri!r}")
    version = MlflowClient().get_model_version_by_alias(m["name"], m["alias"]).version
    local = mlflow.artifacts.download_artifacts(artifact_uri=uri, dst_path=tempfile.mkdtemp(prefix="model-"))
    return Path(local), str(version)


class Models:
    def __init__(self, anomaly: AnomalyModel, breach: BreachModel) -> None:
        self.anomaly, self.breach = anomaly, breach

    @classmethod
    def load(cls, settings: Settings) -> Models:
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        a_dir, a_ver = _fetch(settings.anomaly_model_uri)
        b_dir, b_ver = _fetch(settings.breach_model_uri)
        return cls(AnomalyModel.from_dir(a_dir, a_ver), BreachModel.from_dir(b_dir, b_ver))
