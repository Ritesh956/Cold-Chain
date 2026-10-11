import pytest
from coldchain_ml_api.app import create_app
from coldchain_ml_api.models import AnomalyModel, BreachModel
from coldchain_ml_api.store import Models
from fastapi.testclient import TestClient
from mlapi_helpers import Sim, anomaly_config, breach_model_and_config


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
