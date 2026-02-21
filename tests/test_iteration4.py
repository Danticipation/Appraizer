import asyncio
import json
from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient

from service.auth import APIKeyAuth, RateLimiter
from service.logging_config import JSONFormatter, configure_logging, correlation_id
from service.storage import PostgresStore, RedisCache
from service.valuations import (
    ManheimMMRClient,
    ThirdPartyAuctionClient,
    ValuationsFacade,
    ValuationResult,
)


@pytest.fixture
def store():
    s = PostgresStore("postgresql+asyncpg://bad:bad@127.0.0.1:1/bad")
    s._fallback_mode = True
    s.engine = None
    return s


# ── Valuations Client Tests ─────────────────────────────────────────


@pytest.mark.asyncio
async def test_mmr_client_disabled():
    client = ManheimMMRClient(api_key="")
    result = await client.get_valuation("1HGCM82633A123456")
    assert result.error == "mmr_not_configured"
    assert result.wholesale_value is None


@pytest.mark.asyncio
async def test_mmr_client_batch_disabled():
    client = ManheimMMRClient(api_key="")
    results = await client.get_valuations_batch(["VIN1", "VIN2"])
    assert len(results) == 2
    assert all(r.error == "mmr_not_configured" for r in results)


@pytest.mark.asyncio
async def test_mmr_client_network_failure():
    client = ManheimMMRClient(api_key="test-key", base_url="http://127.0.0.1:1")
    result = await client.get_valuation("1HGCM82633A123456")
    assert result.error is not None
    assert result.wholesale_value is None


@pytest.mark.asyncio
async def test_third_party_disabled():
    client = ThirdPartyAuctionClient(api_key="")
    result = await client.get_sold_price("1HGCM82633A123456")
    assert result.error == "third_party_not_configured"


@pytest.mark.asyncio
async def test_third_party_network_failure():
    client = ThirdPartyAuctionClient(api_key="test-key", base_url="http://127.0.0.1:1")
    result = await client.get_sold_price("1HGCM82633A123456")
    assert result.error is not None


@pytest.mark.asyncio
async def test_valuations_facade_fallback_chain():
    mmr = ManheimMMRClient(api_key="")
    tp = ThirdPartyAuctionClient(api_key="")
    facade = ValuationsFacade(mmr_client=mmr, third_party_client=tp)
    result = await facade.get_actual_price("1HGCM82633A123456")
    assert result.source == "none"
    assert result.error == "no_source_available"


@pytest.mark.asyncio
async def test_valuations_facade_batch():
    mmr = ManheimMMRClient(api_key="")
    tp = ThirdPartyAuctionClient(api_key="")
    facade = ValuationsFacade(mmr_client=mmr, third_party_client=tp)
    results = await facade.get_actual_prices_batch(["VIN1", "VIN2", "VIN3"])
    assert len(results) == 3


# ── Auth Tests ───────────────────────────────────────────────────────


def test_auth_disabled():
    auth = APIKeyAuth(allowed_keys=None)
    assert auth.validate(None) is True
    assert auth.validate("anything") is True


def test_auth_enabled_valid():
    auth = APIKeyAuth(allowed_keys=["my-secret-key"])
    assert auth.validate("my-secret-key") is True


def test_auth_enabled_invalid():
    auth = APIKeyAuth(allowed_keys=["my-secret-key"])
    assert auth.validate("wrong-key") is False
    assert auth.validate(None) is False
    assert auth.validate("") is False


def test_auth_multiple_keys():
    auth = APIKeyAuth(allowed_keys=["key1", "key2", "key3"])
    assert auth.validate("key1") is True
    assert auth.validate("key3") is True
    assert auth.validate("key4") is False


# ── Rate Limiter Tests ───────────────────────────────────────────────


def test_rate_limiter_allows_under_limit():
    limiter = RateLimiter(requests_per_minute=5)
    for _ in range(5):
        assert limiter.check("192.168.1.1") is True


def test_rate_limiter_blocks_over_limit():
    limiter = RateLimiter(requests_per_minute=3)
    for _ in range(3):
        limiter.check("10.0.0.1")
    assert limiter.check("10.0.0.1") is False


def test_rate_limiter_per_ip():
    limiter = RateLimiter(requests_per_minute=2)
    limiter.check("10.0.0.1")
    limiter.check("10.0.0.1")
    assert limiter.check("10.0.0.1") is False
    assert limiter.check("10.0.0.2") is True


def test_rate_limiter_disabled():
    limiter = RateLimiter(requests_per_minute=0)
    assert limiter.check("any") is True


# ── Logging Tests ────────────────────────────────────────────────────


def test_json_formatter():
    import logging
    formatter = JSONFormatter()
    record = logging.LogRecord("test", logging.INFO, "", 0, "hello world", (), None)
    output = formatter.format(record)
    parsed = json.loads(output)
    assert parsed["message"] == "hello world"
    assert parsed["level"] == "INFO"
    assert "timestamp" in parsed


def test_configure_logging():
    configure_logging(level="DEBUG", fmt="text")
    import logging
    root = logging.getLogger()
    assert root.level == logging.DEBUG


def test_correlation_id():
    correlation_id.set("test-123")
    assert correlation_id.get() == "test-123"
    correlation_id.set("")


# ── Feedback Endpoint Tests ──────────────────────────────────────────


@pytest.mark.asyncio
async def test_feedback_storage_round_trip(store):
    appraisal_id = await store.insert_appraisal({
        "vin": "1HGCM82633A123456",
        "input_json": {"mileage": 50000},
        "predicted_value": 12000.0,
        "confidence": 0.92,
        "interval_low": 10800.0,
        "interval_high": 13200.0,
        "population_segment": "normal",
        "region": "US-CA",
        "segment": "sedan",
        "model_version": "test-v1",
    })
    ap = await store.get_appraisal_by_id(appraisal_id)
    assert ap is not None
    assert ap["vin"] == "1HGCM82633A123456"

    fb_id = await store.insert_feedback_actual(
        vin=ap["vin"],
        region=ap["region"],
        segment=ap["segment"],
        actual_close_price=11800.0,
        appraisal_id=appraisal_id,
        source="manual",
    )
    assert fb_id


@pytest.mark.asyncio
async def test_get_appraisal_by_id_not_found(store):
    result = await store.get_appraisal_by_id("nonexistent")
    assert result is None


# ── API Integration Tests ────────────────────────────────────────────


def _make_app(monkeypatch, api_keys: str = ""):
    monkeypatch.setenv("POSTGRES_DSN", "postgresql+asyncpg://bad:bad@127.0.0.1:1/bad")
    monkeypatch.setenv("MODEL_ARTIFACT_DIR", "./artifacts-test")
    monkeypatch.setenv("API_KEYS", api_keys)
    monkeypatch.setenv("RATE_LIMIT_RPM", "0")
    monkeypatch.setenv("LOG_FORMAT", "text")
    from service.api import create_app
    return create_app()


def test_api_no_auth(monkeypatch):
    app = _make_app(monkeypatch, api_keys="")
    with TestClient(app) as client:
        resp = client.get("/health")
        assert resp.status_code == 200

        payload = {
            "vin": "1HGCM82633A123456",
            "region": "US-CA", "segment": "sedan", "title_status": "clean",
            "mileage": 54000, "year": 2020,
            "image_damage_score": 0.12, "image_tamper_score": 0.09,
            "obd_health_score": 0.88, "obd_weighted_dtc_severity": 0.11,
        }
        resp = client.post("/appraise", json=payload)
        assert resp.status_code == 200


def test_api_with_auth(monkeypatch):
    app = _make_app(monkeypatch, api_keys="test-secret-key")
    with TestClient(app) as client:
        payload = {
            "vin": "1HGCM82633A123456",
            "region": "US-CA", "segment": "sedan", "title_status": "clean",
            "mileage": 54000, "year": 2020,
            "image_damage_score": 0.12, "image_tamper_score": 0.09,
            "obd_health_score": 0.88, "obd_weighted_dtc_severity": 0.11,
        }
        resp = client.post("/appraise", json=payload)
        assert resp.status_code == 401

        resp = client.post(
            "/appraise", json=payload,
            headers={"X-API-Key": "test-secret-key"},
        )
        assert resp.status_code == 200


def test_api_prometheus_metrics(monkeypatch):
    app = _make_app(monkeypatch)
    with TestClient(app) as client:
        resp = client.get("/metrics/prometheus")
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/plain")


def test_api_feedback_endpoint(monkeypatch):
    app = _make_app(monkeypatch)
    with TestClient(app) as client:
        payload = {
            "vin": "1HGCM82633A123456",
            "region": "US-CA", "segment": "sedan", "title_status": "clean",
            "mileage": 54000, "year": 2020,
            "image_damage_score": 0.12, "image_tamper_score": 0.09,
            "obd_health_score": 0.88, "obd_weighted_dtc_severity": 0.11,
        }
        appraise_resp = client.post("/appraise", json=payload)
        assert appraise_resp.status_code == 200

        recent = client.get("/appraisals/recent?limit=1").json()
        assert recent["count"] >= 1
        appraisal_id = recent["appraisals"][0]["id"]

        fb_resp = client.post(
            f"/appraisals/{appraisal_id}/feedback",
            json={"actual_close_price": 11500.0, "source": "manual_test"},
        )
        assert fb_resp.status_code == 200
        assert fb_resp.json()["status"] == "recorded"


def test_api_feedback_not_found(monkeypatch):
    app = _make_app(monkeypatch)
    with TestClient(app) as client:
        resp = client.post(
            "/appraisals/nonexistent-id/feedback",
            json={"actual_close_price": 10000.0},
        )
        assert resp.status_code == 404


def test_api_correlation_id(monkeypatch):
    app = _make_app(monkeypatch)
    with TestClient(app) as client:
        resp = client.get("/health", headers={"X-Correlation-ID": "my-trace-123"})
        assert resp.status_code == 200
        assert resp.headers.get("X-Correlation-ID") == "my-trace-123"

        resp2 = client.get("/health")
        assert "X-Correlation-ID" in resp2.headers


def test_api_shadow_ingest_endpoint(monkeypatch):
    app = _make_app(monkeypatch)
    with TestClient(app) as client:
        resp = client.post("/shadow/ingest?window_days=7")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] in ("ok", "skipped")


def test_api_new_endpoints_present(monkeypatch):
    app = _make_app(monkeypatch)
    with TestClient(app) as client:
        for path in ["/metrics", "/metrics/prometheus", "/models", "/appraisals/recent", "/shadow/latest", "/health", "/healthz"]:
            resp = client.get(path)
            assert resp.status_code == 200, f"GET {path} failed with {resp.status_code}"


# ── Shadow Evaluator with Valuations ─────────────────────────────────


@pytest.mark.asyncio
async def test_shadow_ingest_no_client(store):
    from service.shadow_evaluator import ingest_actuals_from_feed
    result = await ingest_actuals_from_feed(store=store, valuations=None)
    assert result["status"] == "skipped"


@pytest.mark.asyncio
async def test_shadow_ingest_no_appraisals(store):
    from service.shadow_evaluator import ingest_actuals_from_feed
    mmr = ManheimMMRClient(api_key="")
    tp = ThirdPartyAuctionClient(api_key="")
    facade = ValuationsFacade(mmr_client=mmr, third_party_client=tp)
    result = await ingest_actuals_from_feed(store=store, valuations=facade)
    assert result["status"] == "skipped"
