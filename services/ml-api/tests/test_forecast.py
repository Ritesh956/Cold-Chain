import numpy as np
import pandas as pd
import pytest
from coldchain_ml.forecast.features import history_frame
from coldchain_ml_api.app import create_app
from coldchain_ml_api.store import Models
from fastapi.testclient import TestClient
from mlapi_helpers import make_forecast_model


@pytest.fixture(scope="module")
def fm():
    return make_forecast_model()


@pytest.fixture()
def forecast_client(models, fm):
    m = Models(models.anomaly, models.breach, None, fm[0])
    with TestClient(create_app(models=m)) as c:
        yield c


def _body(units, start, lengths=(120, 90, 200), end=250):
    as_of = (start + pd.Timedelta(days=end)).date().isoformat()
    items = [
        {"item_id": f"I{i}", "units": units[i, end + 1 - n : end + 1].tolist()} for i, n in enumerate(lengths)
    ]
    return {"as_of": as_of, "items": items}


def test_served_forecast_is_the_training_feature_path_exactly(forecast_client, fm):
    model, units, start = fm
    r = forecast_client.post("/v1/forecast/outflow", json=_body(units, start))
    assert r.status_code == 200
    body = r.json()
    assert body["model"] == {"name": "coldchain-forecast", "version": "test"}
    assert (
        body["horizon_days"] == 28
        and body["first_day"] == (start + pd.Timedelta(days=251)).date().isoformat()
    )
    for i, item in enumerate(body["items"]):
        frame = history_frame(
            units[[i], 250 - 89 : 251], start + pd.Timedelta(days=250)
        )  # only 90 days matter
        want = np.clip(model.model.predict(frame[model.features]), 0, None)
        np.testing.assert_allclose(item["daily_units"], want, rtol=1e-9)
        assert item["total_units"] == pytest.approx(sum(item["daily_units"]))
        assert len(item["daily_units"]) == 28 and min(item["daily_units"]) >= 0


def test_longer_history_gives_the_same_forecast_as_its_last_90_days(forecast_client, fm):
    _, units, start = fm
    a = forecast_client.post("/v1/forecast/outflow", json=_body(units, start, lengths=(90,))).json()
    b = forecast_client.post("/v1/forecast/outflow", json=_body(units, start, lengths=(300 - 49,))).json()
    assert a["items"][0]["daily_units"] == b["items"][0]["daily_units"]


def test_the_date_matters_and_a_dead_item_gets_a_finite_forecast(forecast_client, fm):
    _, units, start = fm
    base = _body(units, start, lengths=(90,))
    shifted = {**base, "as_of": (pd.Timestamp(base["as_of"]) + pd.Timedelta(days=1)).date().isoformat()}
    a = forecast_client.post("/v1/forecast/outflow", json=base).json()["items"][0]["daily_units"]
    b = forecast_client.post("/v1/forecast/outflow", json=shifted).json()["items"][0]["daily_units"]
    assert a != b  # the same history one day later lands on different weekdays
    dead = {"as_of": base["as_of"], "items": [{"item_id": "dead", "units": [0.0] * 90}]}
    out = forecast_client.post("/v1/forecast/outflow", json=dead).json()["items"][0]["daily_units"]
    assert np.isfinite(out).all() and min(out) >= 0


@pytest.mark.parametrize(
    "mutate,code",
    [
        (lambda b: b["items"][0].update(units=b["items"][0]["units"][-89:]), "history_length"),
        (lambda b: b["items"][0].update(units=[1.0] * 401), "history_length"),
        (lambda b: b["items"].append(dict(b["items"][0])), "duplicate_item"),
    ],
)
def test_unscorable_histories_are_422_with_a_reason(forecast_client, fm, mutate, code):
    _, units, start = fm
    body = _body(units, start, lengths=(120, 90))
    mutate(body)
    r = forecast_client.post("/v1/forecast/outflow", json=body)
    assert r.status_code == 422 and r.json()["error"]["code"] == code


@pytest.mark.parametrize(
    "mutate",
    [
        lambda b: b["items"][0]["units"].__setitem__(0, -1.0),
        lambda b: b["items"][0]["units"].__setitem__(0, "abc"),
        lambda b: b.update(items=[]),
        lambda b: b.update(
            items=[b["items"][0]] * 0 + [{"item_id": f"x{i}", "units": [1.0] * 90} for i in range(201)]
        ),
        lambda b: b.update(as_of="not a date"),
        lambda b: b["items"][0].update(item_id=""),
    ],
)
def test_malformed_requests_are_422(forecast_client, fm, mutate):
    _, units, start = fm
    body = _body(units, start, lengths=(120,))
    mutate(body)
    assert forecast_client.post("/v1/forecast/outflow", json=body).status_code == 422


def test_models_endpoint_lists_the_forecast_model(forecast_client):
    info = forecast_client.get("/v1/models").json()["coldchain-forecast"]
    assert info["version"] == "test" and info["min_history_days"] == 90 and info["horizon_days"] == 28


def test_endpoint_is_503_without_the_model_and_the_rest_still_works(client, fm):
    _, units, start = fm
    r = client.post("/v1/forecast/outflow", json=_body(units, start))
    assert r.status_code == 503 and r.json()["error"]["code"] == "model_unavailable"
    assert client.get("/readyz").json() == {"ready": True}
    assert "coldchain-forecast" not in client.get("/v1/models").json()
