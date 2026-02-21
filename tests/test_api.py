from fastapi.testclient import TestClient

from service.api import create_app


def test_appraise_endpoint(monkeypatch):
    monkeypatch.setenv("POSTGRES_DSN", "postgresql+asyncpg://bad:bad@127.0.0.1:1/bad")
    monkeypatch.setenv("NHTSA_BASE_URL", "http://127.0.0.1:1")
    monkeypatch.setenv("MODEL_ARTIFACT_DIR", "./artifacts-test")
    app = create_app()
    with TestClient(app) as client:
        payload = {
            "vin": "1HGCM82633A123456",
            "region": "US-CA",
            "segment": "sedan",
            "title_status": "clean",
            "has_rare_modification": False,
            "mileage": 54000,
            "year": 2020,
            "image_damage_score": 0.12,
            "image_tamper_score": 0.09,
            "obd_health_score": 0.88,
            "obd_weighted_dtc_severity": 0.11,
        }
        resp = client.post("/appraise", json=payload)
        assert resp.status_code == 200
        body = resp.json()
        assert body["appraised_value_usd"] > 0
        assert 0 <= body["confidence"] <= 1
        assert body["interval_low_usd"] <= body["interval_high_usd"]

