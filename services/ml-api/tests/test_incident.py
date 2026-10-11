import pytest
from coldchain_ml_api.app import create_app
from coldchain_ml_api.store import Models
from fastapi.testclient import TestClient
from mlapi_helpers import make_incident_model


@pytest.fixture()
def incident_client(models):
    m = Models(models.anomaly, models.breach, make_incident_model())
    with TestClient(create_app(models=m)) as c:
        yield c


def test_classifies_a_batch_and_scores_sum_to_one(incident_client):
    texts = [
        "pallet crushed at dock 3",
        "temperature alarm freezer 2",
        "truck late again",
        "worker slipped injury",
    ]
    body = incident_client.post("/v1/incidents/classify", json={"texts": texts}).json()
    assert [r["label"] for r in body["results"]] == ["damage", "temperature", "delay", "safety"]
    for r in body["results"]:
        assert sum(r["scores"].values()) == pytest.approx(1.0)
        assert r["confidence"] == pytest.approx(max(r["scores"].values()))
        assert r["needs_review"] is False
    assert body["model"] == {"name": "coldchain-incident", "version": "test"}


def test_unrecognised_text_is_flagged_for_review(incident_client):
    r = incident_client.post("/v1/incidents/classify", json={"texts": ["nothing useful here"]}).json()[
        "results"
    ][0]
    assert r["confidence"] == pytest.approx(0.25) and r["needs_review"] is True  # uniform over 4 classes


@pytest.mark.parametrize(
    "texts",
    [[], [""], ["   "], ["x" * 201], ["ok"] * 65],
)
def test_invalid_inputs_are_422(incident_client, texts):
    assert incident_client.post("/v1/incidents/classify", json={"texts": texts}).status_code == 422


def test_endpoint_is_503_when_the_model_is_not_loaded_but_other_models_still_work(client, sim):
    r = client.post("/v1/incidents/classify", json={"texts": ["pallet crushed"]})
    assert r.status_code == 503 and r.json()["error"]["code"] == "model_unavailable"
    assert client.get("/readyz").json() == {"ready": True}
    assert "coldchain-incident" not in client.get("/v1/models").json()
    assert client.post("/v1/anomaly/score", json=sim.payload(5000)).status_code == 200


def test_model_info_and_metrics_include_the_incident_model(incident_client):
    incident_client.post("/v1/incidents/classify", json={"texts": ["pallet crushed", "late truck"]})
    assert incident_client.get("/v1/models").json()["coldchain-incident"]["version"] == "test"
    text = incident_client.get("/metrics").text
    assert 'model_predictions_total{model="coldchain-incident",outcome="damage"} 1.0' in text
    assert 'model_info{model="coldchain-incident",version="test"} 1.0' in text
    assert "incident_confidence_bucket" in text


def test_a_reports_result_does_not_depend_on_the_other_reports_in_the_request():
    """Regression: dynamic int8 quantisation couples the rows of a batch (measured ~2.7% label flips).
    The model must score each report alone, so its answer cannot depend on its neighbours."""
    import numpy as np
    from mlapi_helpers import StubSession, make_incident_model

    class BatchCoupled(StubSession):
        """Mimics dynamic int8: every row's logits shift with the largest value in the whole batch."""

        def run(self, names, feed):
            logits = super().run(names, feed)[0]
            return [logits - feed["input_ids"].max() * np.array([0.0, 2.0, 0.0, 0.0])]

    m = make_incident_model()
    m.session = BatchCoupled()
    alone = m.classify(["temperature alarm"])[0]
    with_neighbours = m.classify(["temperature alarm", "a very long unrelated sentence " * 3, "injury"])[0]
    assert with_neighbours == alone
