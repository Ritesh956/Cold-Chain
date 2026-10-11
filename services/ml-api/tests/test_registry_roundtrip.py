"""Register models in a throwaway MLflow store and load them the way the service does at startup."""

import json

import mlflow
import mlflow.lightgbm
import numpy as np
import pytest
from coldchain_ml_api.config import Settings
from coldchain_ml_api.store import Models
from conftest import anomaly_config, breach_model_and_config
from mlflow import MlflowClient


@pytest.fixture()
def registry(tmp_path, sim):
    mlflow.set_tracking_uri(f"file:{tmp_path / 'mlruns'}")
    mlflow.set_experiment("t")
    clf, bconf = breach_model_and_config(sim)
    with mlflow.start_run() as run:
        d = tmp_path / "detector.json"
        d.write_text(json.dumps(anomaly_config(sim)), encoding="utf-8")
        mlflow.log_artifact(str(d), "detector")
        mlflow.lightgbm.log_model(clf, "model")
        mlflow.log_dict(bconf, "model/serving_config.json")
        c = MlflowClient()
        for name, path in (("coldchain-anomaly", "detector"), ("coldchain-breach", "model")):
            v = mlflow.register_model(f"runs:/{run.info.run_id}/{path}", name)
            c.set_registered_model_alias(name, "champion", v.version)
    return Settings(mlflow_tracking_uri=f"file:{tmp_path / 'mlruns'}"), clf


def test_models_load_from_registry_alias_and_predict_identically(registry, sim):
    settings, clf = registry
    models = Models.load(settings)
    assert (models.anomaly.version, models.breach.version) == ("1", "1")
    assert models.anomaly.threshold == 4.0 and models.breach.threshold == 0.3

    w = sim.readings.iloc[3000 - 1499 : 3001].reset_index(drop=True)
    from coldchain_ml.anomaly.features import build_features
    from coldchain_ml.breach.data import build_breach_frame

    want = clf.predict_proba(build_breach_frame(w, build_features(w))[models.breach.features])[:, 1]
    got = models.breach.predict(w).risk
    np.testing.assert_allclose(got, want, rtol=1e-12)


def test_moving_the_alias_changes_the_served_version(registry, sim):
    settings, _ = registry
    c = MlflowClient()
    mv = c.get_model_version_by_alias("coldchain-anomaly", "champion")
    v2 = mlflow.register_model(mv.source, "coldchain-anomaly")  # a second version of the same artifact
    assert str(v2.version) == "2"
    assert Models.load(settings).anomaly.version == "1"  # champion still points at v1
    c.set_registered_model_alias("coldchain-anomaly", "champion", str(v2.version))
    assert Models.load(settings).anomaly.version == "2"  # rollout/rollback = move the alias
