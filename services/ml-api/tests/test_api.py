import numpy as np
import pandas as pd
import pytest
from coldchain_ml.anomaly.events_eval import apply_min_run
from coldchain_ml_api.app import create_app
from coldchain_ml_api.config import Settings
from fastapi.testclient import TestClient


def test_health_ready_and_model_info(client):
    assert client.get("/healthz").json() == {"status": "ok"}
    assert client.get("/readyz").json() == {"ready": True}
    info = client.get("/v1/models").json()
    assert info["coldchain-anomaly"]["version"] == "test"
    assert info["coldchain-breach"]["min_history_min"] == 1500
    assert info["coldchain-breach"]["data"] == "SYNTHETIC"


def test_anomaly_alarm_during_door_left_open_and_quiet_otherwise(client, sim):
    ev = sim.event("door_left_open")
    end = sim.index_at(ev["start"]) + 25  # deep into the event
    body = client.post("/v1/anomaly/score", json=sim.payload(end)).json()
    p = body["points"][0]
    assert p["alarm"] is True and p["score"] > body["threshold"]
    assert set(p["drivers"]) == {"temperature", "compressor current"}
    assert "temperature" in p["summary"]

    # a quiet minute: >1 day from every anomaly
    an = sim.events[sim.events["is_anomaly"]]
    quiet = next(
        i
        for i in range(2000, len(sim.readings), 97)
        if ((an["start"] - sim.readings["ts"].iloc[i]).abs() > pd.Timedelta(days=1)).all()
    )
    q = client.post("/v1/anomaly/score", json=sim.payload(quiet)).json()["points"][0]
    assert q["alarm"] is False and q["summary"] is None


def test_serving_score_equals_batch_evaluation_score(client, sim, models):
    """End-to-end train/serve parity: the HTTP score is the score the evaluation code computes."""
    for i in (3000, 9000, 15000):
        got = client.post("/v1/anomaly/score", json=sim.payload(i)).json()["points"][0]["score"]
        want = models.anomaly.detector.score(sim.feats)[i]
        assert got == pytest.approx(want, rel=1e-9)

        got_b = client.post("/v1/breach/predict", json=sim.payload(i)).json()["points"][0]
        want_b = models.breach.model.predict_proba(sim.frame[models.breach.features])[i, 1]
        if bool(sim.frame["at_risk"].iloc[i]):
            assert got_b["risk_score"] == pytest.approx(want_b, rel=1e-9)
        else:
            assert got_b["risk_score"] is None


def test_alarm_debounce_matches_batch(client, sim, models):
    """A multi-minute response applies the same min-run debounce as the batch code."""
    ev = sim.event("door_left_open", 1)
    end = sim.index_at(ev["start"]) + 40
    body = sim.payload(end, n=1500 + 59, return_last=60)
    pts = client.post("/v1/anomaly/score", json=body).json()["points"]
    assert len(pts) == 60 and [p["ts"] for p in pts] == sorted(p["ts"] for p in pts)
    score = models.anomaly.detector.score(sim.feats)
    raw = np.nan_to_num(score, nan=-np.inf) > models.anomaly.threshold
    batch = apply_min_run(raw, models.anomaly.min_run)
    assert [p["alarm"] for p in pts] == list(batch[end - 59 : end + 1])


def test_breach_state_when_already_above_limit(client, sim):
    ev = sim.event("door_left_open")
    start = sim.index_at(ev["start"])
    temp = sim.readings["temp_c"]
    breached = next(i for i in range(start, start + 60) if temp.iloc[i] > -18)
    p = client.post("/v1/breach/predict", json=sim.payload(breached)).json()["points"][0]
    assert p["state"] == "breached" and p["at_risk"] is False and p["risk_score"] is None
    assert p["alarm"] is False
    assert "above -18 C" in p["summary"]


def test_breach_state_recovering_after_the_temperature_drops_back(client, sim):
    """Back under the limit but within 10 minutes of a breach: not 'normal', and not scored as at-risk."""
    temp = sim.readings["temp_c"]
    start = sim.index_at(sim.event("door_left_open")["start"])
    last_above = max(i for i in range(start, start + 120) if temp.iloc[i] > -18)
    i = next(j for j in range(last_above + 1, last_above + 10) if temp.iloc[j] <= -18)
    p = client.post("/v1/breach/predict", json=sim.payload(i)).json()["points"][0]
    assert p["state"] == "recovering" and p["at_risk"] is False and p["risk_score"] is None


def test_breach_warns_before_crossing(client, sim):
    """Somewhere in the minutes before a door-left-open breach, the model raises an alarm."""
    hit = False
    for k in range(3):
        start = sim.index_at(sim.event("door_left_open", k)["start"])
        pts = client.post(
            "/v1/breach/predict", json=sim.payload(start + 14, n=1500 + 29, return_last=30)
        ).json()["points"]
        hit |= any(p["alarm"] for p in pts)
    assert hit


def test_null_temperature_is_accepted(client, sim):
    body = sim.payload(5000)
    body["readings"][-1]["temp_c"] = None
    r = client.post("/v1/breach/predict", json=body).json()["points"][0]
    assert r["state"] == "no_reading" and r["alarm"] is False


def test_naive_timestamps_are_read_as_utc(client, sim):
    body = sim.payload(5000)
    for r in body["readings"]:
        r["ts"] = r["ts"].replace("+00:00", "")
    assert client.post("/v1/anomaly/score", json=body).status_code == 200


@pytest.mark.parametrize(
    ("n", "mutate", "code"),
    [
        (1500, lambda b: b.update(readings=b["readings"][-100:]), "insufficient_history"),
        # 1500 readings but 2-minute steps
        (3000, lambda b: b.update(readings=b["readings"][::2]), "irregular_timestamps"),
        (1500, lambda b: b["readings"].reverse(), "irregular_timestamps"),
        (1500, lambda b: b.update(return_last=500), "insufficient_history"),
    ],
)
def test_invalid_windows_are_rejected_with_a_code(client, sim, n, mutate, code):
    body = sim.payload(5000, n=n)
    mutate(body)
    r = client.post("/v1/breach/predict", json=body)
    assert r.status_code == 422 and r.json()["error"]["code"] == code


@pytest.mark.parametrize(
    "patch",
    [{"temp_c": 500.0}, {"compressor_current_a": -1.0}, {"compressor_on": "maybe"}],
)
def test_out_of_range_values_fail_schema_validation(client, sim, patch):
    body = sim.payload(5000)
    body["readings"][10].update(patch)
    assert client.post("/v1/anomaly/score", json=body).status_code == 422


def test_empty_zone_and_missing_fields_rejected(client, sim):
    body = sim.payload(5000)
    body["zone"] = ""
    assert client.post("/v1/anomaly/score", json=body).status_code == 422
    assert client.post("/v1/anomaly/score", json={"zone": "A"}).status_code == 422


def test_prometheus_metrics_reflect_traffic(client, sim):
    client.post("/v1/anomaly/score", json=sim.payload(5000))
    client.post("/v1/breach/predict", json=sim.payload(5000))
    bad = sim.payload(5000)
    bad["readings"] = bad["readings"][-10:]
    client.post("/v1/anomaly/score", json=bad)
    text = client.get("/metrics").text
    assert 'model_info{model="coldchain-anomaly",version="test"} 1.0' in text
    assert 'input_rejected_total{reason="insufficient_history"} 1.0' in text
    assert 'http_requests_total{method="POST",route="/v1/anomaly/score",status="200"} 1.0' in text
    assert 'http_requests_total{method="POST",route="/v1/anomaly/score",status="422"} 1.0' in text
    assert "model_inference_seconds_count" in text and "anomaly_score_bucket" in text


def test_service_fails_fast_when_models_cannot_load(tmp_path):
    bad = Settings(mlflow_tracking_uri=f"file:{tmp_path}", anomaly_model_uri="models:/nope@champion")
    with pytest.raises(Exception), TestClient(create_app(settings=bad)):  # noqa: B017
        pass


def test_v1_needs_the_api_key_when_one_is_configured(models, sim):
    payload = sim.payload(2000)
    with TestClient(create_app(models=models, settings=Settings(api_key="k"))) as c:
        assert c.get("/healthz").status_code == 200 and c.get("/readyz").status_code == 200
        assert c.get("/metrics").status_code == 200
        assert c.get("/v1/models").status_code == 401
        assert c.post("/v1/anomaly/score", json=payload).status_code == 401
        assert c.get("/v1/models", headers={"X-API-Key": "k"}).status_code == 200
        bearer = {"Authorization": "Bearer k"}
        assert c.post("/v1/anomaly/score", json=payload, headers=bearer).status_code == 200


def test_power_cut_is_described_in_minutes_when_the_off_rule_is_served(models, sim):
    from coldchain_ml_api.models import AnomalyModel
    from coldchain_ml_api.store import Models
    from mlapi_helpers import anomaly_config

    cfg = anomaly_config(sim, threshold=5.7, signals=("temp", "cur_on_360", "off_run"))
    served = Models(AnomalyModel(cfg, "test-off"), models.breach)
    cut = sim.event("power_cut")
    end = sim.index_at(cut["start"]) + 100  # 100 minutes into the cut: compressor off the whole time
    with TestClient(create_app(models=served)) as c:
        p = c.post("/v1/anomaly/score", json=sim.payload(end, n=400)).json()["points"][0]
    assert p["alarm"] is True
    assert "compressor-off duration" in p["drivers"]
    assert "compressor has been off for about" in p["summary"]
    minutes = int(p["summary"].split("about ")[1].split(" minutes")[0])
    assert 95 <= minutes <= 101  # the scaled-back figure is the actual run length
