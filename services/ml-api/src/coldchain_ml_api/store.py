"""Load models from the MLflow registry by URI (models:/<name>@<alias>)."""

from __future__ import annotations

import logging
import re
import tempfile
from pathlib import Path

import mlflow
from coldchain_ml.nlp.serving import IncidentModel
from mlflow import MlflowClient

from coldchain_ml_api.config import Settings
from coldchain_ml_api.models import AnomalyModel, BreachModel

log = logging.getLogger("coldchain_ml_api")

_URI = re.compile(r"^models:/(?P<name>[^@/]+)@(?P<alias>[^@/]+)$")


def _fetch(uri: str) -> tuple[Path, str]:
    m = _URI.match(uri)
    if not m:
        raise ValueError(f"expected models:/<name>@<alias>, got {uri!r}")
    version = MlflowClient().get_model_version_by_alias(m["name"], m["alias"]).version
    local = mlflow.artifacts.download_artifacts(artifact_uri=uri, dst_path=tempfile.mkdtemp(prefix="model-"))
    return Path(local), str(version)


class Models:
    def __init__(
        self, anomaly: AnomalyModel, breach: BreachModel, incident: IncidentModel | None = None
    ) -> None:
        self.anomaly, self.breach, self.incident = anomaly, breach, incident

    @classmethod
    def load(cls, settings: Settings) -> Models:
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        a_dir, a_ver = _fetch(settings.anomaly_model_uri)
        b_dir, b_ver = _fetch(settings.breach_model_uri)
        incident = None
        try:
            i_dir, i_ver = _fetch(settings.incident_model_uri)
            incident = IncidentModel.from_dir(i_dir, i_ver)
        except Exception as e:  # optional model: log and carry on without it
            log.warning("incident model not loaded (%s); /v1/incidents/classify will return 503", e)
        return cls(AnomalyModel.from_dir(a_dir, a_ver), BreachModel.from_dir(b_dir, b_ver), incident)
