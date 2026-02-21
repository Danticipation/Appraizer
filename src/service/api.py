from __future__ import annotations

import asyncio
import time
import uuid
from collections import defaultdict
from contextlib import asynccontextmanager
from typing import Any

import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Request, Response, status
from pydantic import BaseModel, Field

from appraisal.config import TrainingConfig
from appraisal.feature_engineering import infer_population
from appraisal.routing import ManualReviewRouter
from service.auth import APIKeyAuth, RateLimiter
from service.canary_promoter import evaluate_canary, promote_model, start_canary
from service.inference import InferenceFacade
from service.logging_config import configure_logging, correlation_id
from service.messaging import (
    APPRAISAL_REQUESTS_TOPIC,
    APPRAISAL_RESULTS_TOPIC,
    MANUAL_REVIEW_TOPIC,
    RETRAINING_TRIGGERS_TOPIC,
    KafkaBus,
)
from service.retraining import run_retraining
from service.settings import ServiceSettings
from service.shadow_evaluator import run_shadow_evaluation
from service.storage import PostgresStore, RedisCache
from service.valuations import ManheimMMRClient, ThirdPartyAuctionClient, ValuationsFacade
from service.vin import VinDecoder


# ── Request / Response Models ───────────────────────────────────────

class AppraiseRequest(BaseModel):
    vin: str = Field(min_length=17, max_length=17)
    region: str
    segment: str
    title_status: str
    has_rare_modification: bool = False
    mileage: int = Field(ge=0)
    year: int = Field(ge=1980, le=2035)
    image_damage_score: float = Field(ge=0, le=1)
    image_tamper_score: float = Field(ge=0, le=1)
    obd_health_score: float = Field(ge=0, le=1)
    obd_weighted_dtc_severity: float = Field(ge=0, le=1)


class AppraiseResponse(BaseModel):
    appraised_value_usd: float
    confidence: float
    interval_low_usd: float
    interval_high_usd: float
    reject_for_manual_review: bool


class HealthResponse(BaseModel):
    status: str


class ReadinessResponse(BaseModel):
    status: str
    checks: dict[str, bool]


class RetrainResponse(BaseModel):
    scheduled: bool
    message: str


class PromoteRequest(BaseModel):
    version: str
    segment: str
    region: str


class CanaryStartRequest(BaseModel):
    candidate_version: str
    baseline_version: str
    segment: str
    region: str
    traffic_pct: float = 0.10


class FeedbackRequest(BaseModel):
    actual_close_price: float = Field(gt=0)
    source: str = "manual"


# ── Prometheus-style Metrics ────────────────────────────────────────

_prom_counters: dict[str, int] = defaultdict(int)
_prom_histograms: dict[str, list[float]] = defaultdict(list)


def _record_latency(name: str, seconds: float) -> None:
    _prom_histograms[name].append(seconds)
    _prom_counters[f"{name}_count"] += 1


def _prometheus_text() -> str:
    """Render metrics in Prometheus exposition format."""
    lines: list[str] = []
    for k, v in sorted(_prom_counters.items()):
        safe = k.replace(".", "_").replace("-", "_")
        lines.append(f"# TYPE appraisal_{safe} counter")
        lines.append(f"appraisal_{safe} {v}")

    for name, vals in sorted(_prom_histograms.items()):
        if not vals:
            continue
        safe = name.replace(".", "_").replace("-", "_")
        sorted_vals = sorted(vals)
        n = len(sorted_vals)
        lines.append(f"# TYPE appraisal_{safe}_seconds summary")
        for q in (0.5, 0.9, 0.95, 0.99):
            idx = min(int(n * q), n - 1)
            lines.append(f'appraisal_{safe}_seconds{{quantile="{q}"}} {sorted_vals[idx]:.6f}')
        lines.append(f"appraisal_{safe}_seconds_count {n}")
        lines.append(f"appraisal_{safe}_seconds_sum {sum(sorted_vals):.6f}")

    return "\n".join(lines) + "\n"


# ── App Factory ─────────────────────────────────────────────────────

def create_app() -> FastAPI:
    settings = ServiceSettings()
    configure_logging(level=settings.log_level, fmt=settings.log_format)

    cfg = TrainingConfig()
    router = ManualReviewRouter(confidence_threshold=settings.confidence_threshold)
    cache = RedisCache(redis_url=settings.redis_url)
    store = PostgresStore(dsn=settings.postgres_dsn)
    kafka = KafkaBus(
        bootstrap_servers=settings.kafka_bootstrap_servers,
        client_id=settings.kafka_client_id,
    )
    inference = InferenceFacade(artifact_dir=settings.model_artifact_dir)
    vin_decoder = VinDecoder(cache=cache, base_url=settings.nhtsa_base_url, ttl_seconds=settings.vin_cache_ttl_seconds)

    mmr = ManheimMMRClient(api_key=settings.mmr_api_key, base_url=settings.mmr_base_url)
    third_party = ThirdPartyAuctionClient(api_key=settings.third_party_auction_api_key, base_url=settings.third_party_auction_base_url)
    valuations = ValuationsFacade(mmr_client=mmr, third_party_client=third_party)

    api_keys = [k.strip() for k in settings.api_keys.split(",") if k.strip()] if settings.api_keys else []
    auth = APIKeyAuth(allowed_keys=api_keys or None)
    limiter = RateLimiter(requests_per_minute=settings.rate_limit_rpm)

    stop_event = asyncio.Event()

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        await cache.connect()
        await store.connect()
        await kafka.connect()

        async def _retrain_handler(_: dict[str, Any]) -> None:
            await run_retraining(store=store, artifact_dir=settings.model_artifact_dir, config=cfg)

        consumer_task = asyncio.create_task(kafka.consume_retraining_forever(_retrain_handler, stop_event))
        try:
            yield
        finally:
            stop_event.set()
            consumer_task.cancel()
            await cache.close()
            await store.close()
            await kafka.close()

    app = FastAPI(title="AI Automobile Appraisal API", version="0.4.0", lifespan=lifespan)

    @app.middleware("http")
    async def correlation_id_middleware(request: Request, call_next: Any) -> Response:
        cid = request.headers.get("X-Correlation-ID") or uuid.uuid4().hex[:12]
        correlation_id.set(cid)
        response = await call_next(request)
        response.headers["X-Correlation-ID"] = cid
        return response

    @app.middleware("http")
    async def rate_limit_middleware(request: Request, call_next: Any) -> Response:
        return await limiter.middleware(request, call_next)

    # ── Core Appraisal ──────────────────────────────────────────────

    @app.post("/appraise", response_model=AppraiseResponse)
    async def appraise(payload: AppraiseRequest, _: str | None = Depends(auth)) -> AppraiseResponse:
        t0 = time.monotonic()
        feature_key = f"appraise:{payload.vin}:{payload.region}:{payload.segment}"
        cached = await cache.get_json(feature_key)
        if cached:
            _record_latency("appraise", time.monotonic() - t0)
            _prom_counters["appraise_cache_hit"] += 1
            return AppraiseResponse(**cached)

        vin_info = await vin_decoder.decode(payload.vin)
        frame_input = payload.model_dump()
        if vin_info.get("model_year"):
            frame_input["year"] = vin_info["model_year"]
        frame = pd.DataFrame([frame_input])
        model_output = inference.predict(
            frame=frame,
            region=payload.region,
            segment=payload.segment,
            title_status=payload.title_status,
            has_rare_modification=payload.has_rare_modification,
        )
        if model_output.prediction <= 0:
            raise HTTPException(status_code=500, detail="Invalid model output")

        reject = router.route(payload.model_dump(), model_output.confidence)
        result = {
            "appraised_value_usd": round(model_output.prediction, 2),
            "confidence": round(model_output.confidence, 4),
            "interval_low_usd": round(model_output.interval_low, 2),
            "interval_high_usd": round(model_output.interval_high, 2),
            "reject_for_manual_review": reject,
        }

        population = infer_population(payload.title_status, payload.has_rare_modification)
        await store.insert_appraisal(
            {
                "vin": payload.vin,
                "input_json": payload.model_dump(),
                "vin_json": vin_info,
                "predicted_value": model_output.prediction,
                "confidence": model_output.confidence,
                "interval_low": model_output.interval_low,
                "interval_high": model_output.interval_high,
                "population_segment": population,
                "region": payload.region,
                "segment": payload.segment,
                "model_version": model_output.model_version,
                "reviewed_flag": False,
                "route_to_manual": reject,
            }
        )
        await cache.set_json(feature_key, result, ttl_seconds=settings.appraisal_cache_ttl_seconds)

        await kafka.publish(APPRAISAL_REQUESTS_TOPIC, payload.model_dump(), key=payload.vin)
        await kafka.publish(APPRAISAL_RESULTS_TOPIC, result, key=payload.vin)
        if reject:
            await kafka.publish(
                MANUAL_REVIEW_TOPIC,
                {"vin": payload.vin, "region": payload.region, "segment": payload.segment, "confidence": result["confidence"]},
                key=payload.vin,
            )

        _record_latency("appraise", time.monotonic() - t0)
        _prom_counters[f"confidence_bucket_{'high' if result['confidence'] >= 0.90 else 'low'}"] += 1
        if reject:
            _prom_counters["rejections"] += 1

        return AppraiseResponse(**result)

    # ── Health ──────────────────────────────────────────────────────

    @app.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        return HealthResponse(status="ok")

    @app.get("/healthz", response_model=HealthResponse)
    async def healthz() -> HealthResponse:
        return HealthResponse(status="ok")

    @app.get("/ready", response_model=ReadinessResponse)
    async def ready() -> ReadinessResponse:
        checks = {
            "redis": await cache.ping(),
            "postgres": await store.ping(),
            "kafka": await kafka.ping(),
        }
        all_ready = all(checks.values())
        if not all_ready:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=ReadinessResponse(status="degraded", checks=checks).model_dump(),
            )
        return ReadinessResponse(status="ready", checks=checks)

    # ── Retrain / Shadow / Canary ───────────────────────────────────

    @app.post("/retrain/trigger", response_model=RetrainResponse)
    async def trigger_retrain(_: str | None = Depends(auth)) -> RetrainResponse:
        await kafka.publish(RETRAINING_TRIGGERS_TOPIC, {"source": "api_manual"})
        return RetrainResponse(scheduled=True, message="retraining trigger published")

    @app.post("/shadow/evaluate")
    async def trigger_shadow_evaluation(
        window_days: int = 14, use_mock: bool = False,
        _: str | None = Depends(auth),
    ) -> dict[str, Any]:
        result = await run_shadow_evaluation(
            store=store,
            kafka=kafka,
            valuations=valuations,
            window_days=window_days,
            use_mock_actuals=use_mock,
        )
        return result

    @app.post("/shadow/ingest")
    async def trigger_actuals_ingestion(
        window_days: int = 14, _: str | None = Depends(auth),
    ) -> dict[str, Any]:
        from service.shadow_evaluator import ingest_actuals_from_feed
        return await ingest_actuals_from_feed(store=store, valuations=valuations, window_days=window_days)

    @app.post("/canary/start")
    async def api_canary_start(req: CanaryStartRequest, _: str | None = Depends(auth)) -> dict[str, Any]:
        return await start_canary(
            store=store, cache=cache,
            candidate_version=req.candidate_version,
            baseline_version=req.baseline_version,
            segment=req.segment, region=req.region,
            initial_traffic_pct=req.traffic_pct,
        )

    @app.post("/canary/evaluate")
    async def api_canary_evaluate(req: CanaryStartRequest, _: str | None = Depends(auth)) -> dict[str, Any]:
        return await evaluate_canary(
            store=store, cache=cache, kafka=kafka,
            candidate_version=req.candidate_version,
            baseline_version=req.baseline_version,
            segment=req.segment, region=req.region,
        )

    @app.post("/promote")
    async def api_promote(req: PromoteRequest, _: str | None = Depends(auth)) -> dict[str, Any]:
        return await promote_model(
            store=store, cache=cache, kafka=kafka,
            version=req.version, segment=req.segment, region=req.region,
        )

    # ── Feedback ────────────────────────────────────────────────────

    @app.post("/appraisals/{appraisal_id}/feedback")
    async def submit_feedback(
        appraisal_id: str, req: FeedbackRequest,
        _: str | None = Depends(auth),
    ) -> dict[str, Any]:
        appraisal = await store.get_appraisal_by_id(appraisal_id)
        if appraisal is None:
            raise HTTPException(status_code=404, detail="Appraisal not found")
        fb_id = await store.insert_feedback_actual(
            vin=appraisal["vin"],
            region=appraisal["region"],
            segment=appraisal["segment"],
            actual_close_price=req.actual_close_price,
            appraisal_id=appraisal_id,
            source=req.source,
        )
        return {"feedback_id": fb_id, "appraisal_id": appraisal_id, "status": "recorded"}

    # ── Operational Dashboard Endpoints ─────────────────────────────

    @app.get("/metrics")
    async def get_metrics() -> dict[str, Any]:
        latencies = _prom_histograms.get("appraise", [])
        return {
            "counters": dict(_prom_counters),
            "appraise_latency": {
                "count": len(latencies),
                "p50_ms": round(sorted(latencies)[len(latencies) // 2] * 1000, 1) if latencies else 0,
                "p95_ms": round(sorted(latencies)[int(len(latencies) * 0.95)] * 1000, 1) if latencies else 0,
                "p99_ms": round(sorted(latencies)[int(len(latencies) * 0.99)] * 1000, 1) if latencies else 0,
            },
        }

    @app.get("/metrics/prometheus")
    async def get_prometheus_metrics() -> Response:
        return Response(content=_prometheus_text(), media_type="text/plain; charset=utf-8")

    @app.get("/models")
    async def get_models() -> dict[str, Any]:
        active = await store.get_active_models()
        canaries = await store.get_active_canaries()
        return {"active_models": active, "active_canaries": canaries}

    @app.get("/appraisals/recent")
    async def get_recent_appraisals(limit: int = 20) -> dict[str, Any]:
        rows = await store.get_recent_appraisals(limit=min(limit, 100))
        sanitized = []
        for r in rows:
            entry = {k: v for k, v in r.items() if k != "input_json"}
            for k, v in entry.items():
                if hasattr(v, "isoformat"):
                    entry[k] = v.isoformat()
            sanitized.append(entry)
        return {"count": len(sanitized), "appraisals": sanitized}

    @app.get("/shadow/latest")
    async def get_latest_shadow_metrics(model_version: str = "", limit: int = 10) -> dict[str, Any]:
        if model_version:
            metrics = await store.fetch_shadow_metrics(model_version, limit=limit)
        else:
            metrics = []
            active = await store.get_active_models()
            for m in active[:5]:
                metrics.extend(await store.fetch_shadow_metrics(m["version"], limit=3))
        sanitized = []
        for m in metrics:
            entry = dict(m)
            for k, v in entry.items():
                if hasattr(v, "isoformat"):
                    entry[k] = v.isoformat()
            sanitized.append(entry)
        return {"metrics": sanitized}

    return app


app = create_app()
