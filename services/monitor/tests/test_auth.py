from __future__ import annotations

from coldchain_monitor.app import create_app
from coldchain_monitor.config import Settings
from fastapi.testclient import TestClient


class FakeSource:
    async def zones(self):
        return []

    async def readings(self, zone, minutes):
        return []

    async def ml_metrics(self):
        return ""

    async def close(self):
        return None


def test_drift_endpoints_need_the_key_but_health_and_metrics_do_not():
    app = create_app(FakeSource(), Settings(api_key="k"), run_background=False)
    with TestClient(app) as c:
        assert c.get("/healthz").status_code == 200 and c.get("/metrics").status_code == 200
        assert c.get("/v1/drift").status_code == 401
        assert c.post("/v1/drift/check").status_code == 401
        assert c.get("/v1/drift", headers={"X-API-Key": "k"}).status_code == 200


def test_source_sends_the_service_keys():
    from coldchain_monitor.sources import HttpSource

    s = HttpSource("http://core", "http://ml", core_key="ck", ml_key="mk")
    assert s._core.headers["x-api-key"] == "ck" and s._ml.headers["x-api-key"] == "mk"
    assert "x-api-key" not in HttpSource("http://core", "http://ml")._core.headers
