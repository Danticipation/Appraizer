import asyncio
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from service.canary_promoter import evaluate_canary, promote_model, rollback_canary, start_canary
from service.drift_detector import population_stability_index, run_drift_checks
from service.messaging import KafkaBus
from service.shadow_evaluator import compute_segment_metrics, generate_mock_actuals, run_shadow_evaluation
from service.storage import PostgresStore, RedisCache


@pytest.fixture
def store():
    s = PostgresStore("postgresql+asyncpg://bad:bad@127.0.0.1:1/bad")
    s._fallback_mode = True
    s.engine = None
    return s


@pytest.fixture
def cache():
    c = RedisCache(redis_url="redis://localhost:65535/0")
    return c


@pytest.fixture
def kafka():
    return KafkaBus(bootstrap_servers="localhost:65535", client_id="test")


def _make_appraisal(vin: str, predicted: float, region: str = "US-CA", segment: str = "sedan", version: str = "v1") -> dict:
    return {
        "vin": vin,
        "input_json": {"region": region, "segment": segment, "mileage": 50000, "year": 2020},
        "vin_json": {},
        "predicted_value": predicted,
        "confidence": 0.92,
        "interval_low": predicted * 0.88,
        "interval_high": predicted * 1.12,
        "population_segment": "normal",
        "region": region,
        "segment": segment,
        "model_version": version,
        "reviewed_flag": False,
        "route_to_manual": False,
    }


# ── Shadow Evaluator Tests ──────────────────────────────────────────


def test_compute_segment_metrics():
    df = pd.DataFrame({
        "predicted_value": [10000, 12000, 11000, 9500, 13000],
        "actual_close_price": [10200, 11500, 11200, 10000, 12500],
        "confidence": [0.95, 0.91, 0.88, 0.93, 0.96],
        "route_to_manual": [False, False, True, False, False],
    })
    metrics = compute_segment_metrics(df)
    assert "mae" in metrics
    assert "rmse" in metrics
    assert "bias" in metrics
    assert "coverage_pct" in metrics
    assert "calibration_error" in metrics
    assert metrics["sample_count"] == 5
    assert metrics["coverage_pct"] == 80.0
    assert metrics["mae"] > 0
    assert metrics["rmse"] >= metrics["mae"]


def test_generate_mock_actuals():
    df = pd.DataFrame({
        "predicted_value": [10000, 20000, 30000],
        "route_to_manual": [False, True, False],
    })
    result = generate_mock_actuals(df, noise_pct=0.10)
    assert "actual_close_price" in result.columns
    for _, row in result.iterrows():
        assert abs(row["actual_close_price"] - row["predicted_value"]) <= row["predicted_value"] * 0.10 + 1


@pytest.mark.asyncio
async def test_shadow_evaluation_with_mock_actuals(store):
    for i in range(10):
        await store.insert_appraisal(
            _make_appraisal(f"1HGCM82633A{i:06d}", 10000 + i * 500)
        )

    result = await run_shadow_evaluation(
        store=store, window_days=1, use_mock_actuals=True,
    )
    assert result["status"] == "ok"
    assert result["evaluated_groups"] >= 1
    assert len(result["results"]) >= 1


@pytest.mark.asyncio
async def test_shadow_evaluation_no_data(store):
    result = await run_shadow_evaluation(store=store, window_days=1, use_mock_actuals=False)
    assert result["status"] == "skipped"


@pytest.mark.asyncio
async def test_shadow_evaluation_with_real_actuals(store):
    for i in range(10):
        vin = f"1HGCM82633B{i:06d}"
        await store.insert_appraisal(_make_appraisal(vin, 10000 + i * 500))
        await store.insert_feedback_actual(
            vin=vin, region="US-CA", segment="sedan",
            actual_close_price=10200 + i * 480,
        )

    result = await run_shadow_evaluation(store=store, window_days=1)
    assert result["status"] == "ok"
    assert result["evaluated_groups"] >= 1


# ── Canary Promoter Tests ───────────────────────────────────────────


@pytest.mark.asyncio
async def test_start_canary(store, cache):
    result = await start_canary(
        store=store, cache=cache,
        candidate_version="v2", baseline_version="v1",
        segment="sedan", region="US-CA",
    )
    assert result["status"] == "shadow"
    assert result["traffic_pct"] == 0.10

    canaries = await store.get_active_canaries()
    assert len(canaries) == 1
    assert canaries[0]["candidate_version"] == "v2"


@pytest.mark.asyncio
async def test_promote_model(store, cache):
    await store.insert_model_version(
        segment="sedan", region="US-CA", population_segment="normal",
        version="v2", metrics_json={"mae": 800},
    )
    result = await promote_model(
        store=store, cache=cache,
        version="v2", segment="sedan", region="US-CA", shadow_mae=750.0,
    )
    assert result["action"] == "promoted"

    active = await store.get_active_models()
    assert any(m["version"] == "v2" for m in active)


@pytest.mark.asyncio
async def test_rollback_canary(store, cache):
    await start_canary(
        store=store, cache=cache,
        candidate_version="v3", baseline_version="v1",
        segment="sedan", region="US-CA",
    )
    result = await rollback_canary(
        store=store, cache=cache,
        candidate_version="v3", baseline_version="v1",
        segment="sedan", region="US-CA",
        reason="test regression",
    )
    assert result["action"] == "rolled_back"


@pytest.mark.asyncio
async def test_evaluate_canary_no_metrics(store, cache):
    result = await evaluate_canary(
        store=store, cache=cache,
        candidate_version="v_new", baseline_version="v_old",
        segment="sedan", region="US-CA",
    )
    assert result["action"] == "wait"


@pytest.mark.asyncio
async def test_evaluate_canary_promote_on_improvement(store, cache):
    await store.insert_model_version(
        segment="sedan", region="US-CA", population_segment="normal",
        version="candidate-v1", metrics_json={},
    )
    await store.insert_shadow_metric({
        "model_version": "candidate-v1", "segment": "sedan", "region": "US-CA",
        "population_segment": "normal", "mae": 500, "rmse": 600, "bias": 10,
        "coverage_pct": 95, "calibration_error": 0.02, "sample_count": 100,
        "window_start": datetime.now(timezone.utc) - timedelta(days=7),
        "window_end": datetime.now(timezone.utc),
    })
    await store.insert_shadow_metric({
        "model_version": "baseline-v1", "segment": "sedan", "region": "US-CA",
        "population_segment": "normal", "mae": 1200, "rmse": 1400, "bias": 50,
        "coverage_pct": 90, "calibration_error": 0.05, "sample_count": 100,
        "window_start": datetime.now(timezone.utc) - timedelta(days=7),
        "window_end": datetime.now(timezone.utc),
    })
    await start_canary(
        store=store, cache=cache,
        candidate_version="candidate-v1", baseline_version="baseline-v1",
        segment="sedan", region="US-CA", initial_traffic_pct=0.90,
    )
    result = await evaluate_canary(
        store=store, cache=cache,
        candidate_version="candidate-v1", baseline_version="baseline-v1",
        segment="sedan", region="US-CA",
    )
    assert result["action"] == "promoted"


# ── Drift Detector Tests ────────────────────────────────────────────


def test_psi_identical_distributions():
    rng = np.random.RandomState(0)
    a = rng.normal(50000, 5000, 500)
    psi = population_stability_index(a, a)
    assert psi < 0.05


def test_psi_shifted_distribution():
    rng = np.random.RandomState(0)
    ref = rng.normal(50000, 5000, 500)
    cur = rng.normal(60000, 5000, 500)
    psi = population_stability_index(ref, cur)
    assert psi > 0.20


def test_run_drift_checks_no_drift():
    rng1 = np.random.RandomState(42)
    rng2 = np.random.RandomState(42)
    n = 500
    ref = pd.DataFrame({"mileage": rng1.normal(60000, 10000, n), "year": rng1.normal(2018, 3, n)})
    cur = pd.DataFrame({"mileage": rng2.normal(60000, 10000, n), "year": rng2.normal(2018, 3, n)})
    result = run_drift_checks(ref, cur, feature_cols=["mileage", "year"])
    assert result["features_checked"] == 2
    assert not result["drift_detected"]


def test_run_drift_checks_with_drift():
    rng = np.random.RandomState(42)
    n = 200
    ref = pd.DataFrame({"mileage": rng.normal(50000, 5000, n)})
    cur = pd.DataFrame({"mileage": rng.normal(80000, 5000, n)})
    result = run_drift_checks(ref, cur, feature_cols=["mileage"])
    assert result["drift_detected"]
    assert len(result["drifted_features"]) == 1
    assert result["drifted_features"][0]["feature"] == "mileage"


# ── VIN Batch Decode Test ───────────────────────────────────────────


@pytest.mark.asyncio
async def test_vin_batch_decode_fallback():
    from service.vin import VinDecoder

    cache = RedisCache(redis_url="redis://localhost:65535/0")
    decoder = VinDecoder(cache=cache, base_url="http://127.0.0.1:1", ttl_seconds=60)
    vins = ["1HGCM82633A123456", "5YJSA1E26HF000001", "WVWZZZ3CZWE123456"]
    results = await decoder.decode_batch(vins)
    assert len(results) == 3
    for r in results:
        assert r["decode_source"] == "fallback"
        assert "model_year" in r


# ── API Endpoint Tests ──────────────────────────────────────────────


def test_new_api_endpoints(monkeypatch):
    monkeypatch.setenv("POSTGRES_DSN", "postgresql+asyncpg://bad:bad@127.0.0.1:1/bad")
    monkeypatch.setenv("MODEL_ARTIFACT_DIR", "./artifacts-test")
    from service.api import create_app
    from fastapi.testclient import TestClient

    app = create_app()
    with TestClient(app) as client:
        resp = client.get("/metrics")
        assert resp.status_code == 200
        body = resp.json()
        assert "counters" in body
        assert "appraise_latency" in body

        resp = client.get("/models")
        assert resp.status_code == 200
        assert "active_models" in resp.json()

        resp = client.get("/appraisals/recent")
        assert resp.status_code == 200
        assert "appraisals" in resp.json()

        resp = client.get("/shadow/latest")
        assert resp.status_code == 200
        assert "metrics" in resp.json()

        resp = client.get("/health")
        assert resp.status_code == 200


# ── Storage Schema Tests ────────────────────────────────────────────


@pytest.mark.asyncio
async def test_shadow_metrics_storage(store):
    row_id = await store.insert_shadow_metric({
        "model_version": "test-v1", "segment": "sedan", "region": "US-CA",
        "population_segment": "normal", "mae": 800, "rmse": 950, "bias": -30,
        "coverage_pct": 92, "calibration_error": 0.03, "sample_count": 50,
        "window_start": datetime.now(timezone.utc) - timedelta(days=7),
        "window_end": datetime.now(timezone.utc),
    })
    assert row_id

    metrics = await store.fetch_shadow_metrics("test-v1")
    assert len(metrics) == 1
    assert metrics[0]["mae"] == 800


@pytest.mark.asyncio
async def test_model_promotion_lifecycle(store):
    await store.insert_model_version(
        segment="suv", region="US-TX", population_segment="normal",
        version="lifecycle-v1", metrics_json={"mae": 900},
    )
    await store.update_model_promotion("lifecycle-v1", promotion_status="promoted", shadow_mae=850.0)
    await store.set_active_model("lifecycle-v1", "suv", "US-TX")

    active = await store.get_active_models()
    assert any(m["version"] == "lifecycle-v1" and m["is_active"] for m in active)
