import asyncio

import pytest
from fastapi.testclient import TestClient

from service.api import create_app
from service.messaging import KafkaBus, RETRAINING_TRIGGERS_TOPIC
from service.storage import PostgresStore, RedisCache


@pytest.mark.asyncio
async def test_storage_round_trip_sqlite():
    store = PostgresStore("postgresql+asyncpg://bad:bad@127.0.0.1:1/bad")
    await store.connect()
    await store.insert_appraisal(
        {
            "vin": "1HGCM82633A123456",
            "input_json": {"region": "US-CA", "segment": "sedan", "mileage": 50000, "year": 2020},
            "vin_json": {"make": "HONDA"},
            "predicted_value": 12000.0,
            "confidence": 0.91,
            "interval_low": 11000.0,
            "interval_high": 13000.0,
            "population_segment": "normal",
            "region": "US-CA",
            "segment": "sedan",
            "model_version": "test-v1",
            "reviewed_flag": False,
            "route_to_manual": True,
        }
    )
    frame = await store.fetch_training_frame(limit=10)
    await store.close()
    assert not frame.empty
    assert "auction_close_price" in frame.columns


@pytest.mark.asyncio
async def test_redis_cache_round_trip_fallback():
    cache = RedisCache(redis_url="redis://localhost:65535/0")
    await cache.connect()
    await cache.set_json("k1", {"ok": True}, ttl_seconds=60)
    out = await cache.get_json("k1")
    await cache.close()
    assert out == {"ok": True}


@pytest.mark.asyncio
async def test_kafka_publish_consume_fallback_queue():
    bus = KafkaBus(bootstrap_servers="localhost:65535", client_id="test-client")
    await bus.connect()
    seen = []
    stop = asyncio.Event()

    async def handler(event):
        seen.append(event)
        stop.set()

    consumer_task = asyncio.create_task(bus.consume_retraining_forever(handler, stop))
    await bus.publish(RETRAINING_TRIGGERS_TOPIC, {"source": "test"})
    await asyncio.wait_for(stop.wait(), timeout=2.0)
    consumer_task.cancel()
    await bus.close()
    assert seen and seen[0]["source"] == "test"


def test_appraise_and_ready_endpoints(monkeypatch):
    monkeypatch.setenv("POSTGRES_DSN", "postgresql+asyncpg://bad:bad@127.0.0.1:1/bad")
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
        appraise_resp = client.post("/appraise", json=payload)
        assert appraise_resp.status_code == 200
        body = appraise_resp.json()
        assert body["reject_for_manual_review"] is True

        health_resp = client.get("/health")
        assert health_resp.status_code == 200

        ready_resp = client.get("/ready")
        assert ready_resp.status_code == 503
