# Service Layer

Async FastAPI service with production-grade adapters, security, and monitoring.

## Modules

| Module | Purpose |
|--------|---------|
| `api.py` | FastAPI app factory, all HTTP endpoints, middleware, Prometheus metrics |
| `auth.py` | API key authentication (SHA-256 hashed) + in-memory rate limiter |
| `storage.py` | PostgreSQL (SQLAlchemy async) + Redis cache + in-memory fallbacks |
| `messaging.py` | Kafka producer/consumer (aiokafka) + in-memory fallback queues |
| `vin.py` | NHTSA VIN decode (single + batch) with Redis caching + year-code fallback |
| `valuations.py` | Manheim MMR + third-party auction API clients with unified facade |
| `inference.py` | Model artifact loader (joblib) + baseline fallback predictor |
| `retraining.py` | Training pipeline runner with drift detection integration |
| `shadow_evaluator.py` | Shadow evaluation + actuals ingestion from feeds |
| `canary_promoter.py` | Canary lifecycle: start → evaluate → ramp → promote / rollback |
| `drift_detector.py` | Feature distribution drift (KS-test + PSI) with Kafka alerting |
| `logging_config.py` | Structured JSON logging + correlation ID propagation |
| `settings.py` | Pydantic settings with env var / `.env` support |

## Database Tables

| Table | Description |
|-------|-------------|
| `appraisals` | All appraisal records with VIN, prediction, confidence, intervals, routing |
| `model_versions` | Model registry: version, metrics, drift data, promotion status, is_active |
| `feedback_actuals` | Realized auction close prices (with optional appraisal_id FK, source tag) |
| `shadow_metrics` | Per-segment shadow evaluation results |
| `canary_state` | Active canary deployments tracking traffic %, status, promotion time |

## Security

- **API key auth**: `X-API-Key` header validated against SHA-256 hashed keys. Disabled when `API_KEYS` is empty.
- **Rate limiting**: Sliding-window per client IP. Configurable via `RATE_LIMIT_RPM` (0 = disabled).
- **Correlation IDs**: `X-Correlation-ID` header propagated through request/response + JSON logs.

## Auction Feed Integration

Fallback chain: Manheim MMR → Third-party API → Mock actuals

1. `ValuationsFacade.get_actual_price(vin)` tries MMR first, then third-party.
2. `ingest_actuals_from_feed()` pulls realized prices for settled appraisals (past settlement window).
3. `run_shadow_evaluation()` auto-calls ingestion before computing metrics.
4. All sources tagged in `feedback_actuals.source` column for traceability.

## Prometheus Metrics

`GET /metrics/prometheus` returns:
- `appraisal_appraise_count` — total appraisals processed
- `appraisal_appraise_cache_hit` — cache hit count
- `appraisal_rejections` — manual review rejections
- `appraisal_confidence_bucket_high/low` — confidence distribution
- `appraisal_appraise_seconds{quantile=...}` — latency percentiles
