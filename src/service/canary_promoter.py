from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from service.messaging import KafkaBus
from service.storage import PostgresStore, RedisCache

logger = logging.getLogger(__name__)

CANARY_ALERT_TOPIC = "canary_events"

DEFAULT_SHADOW_HOURS = 72
DEFAULT_TRAFFIC_STEP = 0.10
MAX_TRAFFIC_PCT = 1.0
MAE_IMPROVEMENT_THRESHOLD = 50.0


async def start_canary(
    *,
    store: PostgresStore,
    cache: RedisCache,
    candidate_version: str,
    baseline_version: str,
    segment: str,
    region: str,
    initial_traffic_pct: float = 0.10,
) -> dict[str, Any]:
    record = {
        "candidate_version": candidate_version,
        "baseline_version": baseline_version,
        "segment": segment,
        "region": region,
        "traffic_pct": initial_traffic_pct,
        "status": "shadow",
        "started_at": datetime.now(timezone.utc),
    }
    canary_id = await store.upsert_canary_state(record)
    await cache.set_json(
        f"canary:{segment}:{region}",
        {"candidate": candidate_version, "baseline": baseline_version, "traffic_pct": initial_traffic_pct},
        ttl_seconds=DEFAULT_SHADOW_HOURS * 3600,
    )
    logger.info("Canary started: %s vs %s for %s/%s at %.0f%%", candidate_version, baseline_version, region, segment, initial_traffic_pct * 100)
    return {"canary_id": canary_id, "status": "shadow", "traffic_pct": initial_traffic_pct}


async def evaluate_canary(
    *,
    store: PostgresStore,
    cache: RedisCache,
    kafka: KafkaBus | None = None,
    candidate_version: str,
    baseline_version: str,
    segment: str,
    region: str,
    mae_threshold: float = MAE_IMPROVEMENT_THRESHOLD,
    traffic_step: float = DEFAULT_TRAFFIC_STEP,
) -> dict[str, Any]:
    candidate_metrics = await store.fetch_shadow_metrics(candidate_version, limit=1)
    baseline_metrics = await store.fetch_shadow_metrics(baseline_version, limit=1)

    if not candidate_metrics:
        return {"action": "wait", "reason": "no_candidate_shadow_metrics"}
    if not baseline_metrics:
        return {"action": "promote", "reason": "no_baseline_metrics_to_compare"}

    c_mae = candidate_metrics[0]["mae"]
    b_mae = baseline_metrics[0]["mae"]
    improvement = b_mae - c_mae

    if improvement >= mae_threshold:
        canaries = await store.get_active_canaries()
        current = next(
            (c for c in canaries if c["candidate_version"] == candidate_version),
            None,
        )
        current_pct = current["traffic_pct"] if current else 0.10

        if current_pct + traffic_step >= MAX_TRAFFIC_PCT:
            return await promote_model(
                store=store, cache=cache, kafka=kafka,
                version=candidate_version, segment=segment, region=region,
                shadow_mae=c_mae,
            )

        new_pct = min(MAX_TRAFFIC_PCT, current_pct + traffic_step)
        await store.upsert_canary_state({
            "candidate_version": candidate_version,
            "baseline_version": baseline_version,
            "segment": segment,
            "region": region,
            "traffic_pct": new_pct,
            "status": "canary",
            "metrics_json": {"candidate_mae": c_mae, "baseline_mae": b_mae, "improvement": improvement},
        })
        await cache.set_json(
            f"canary:{segment}:{region}",
            {"candidate": candidate_version, "baseline": baseline_version, "traffic_pct": new_pct},
            ttl_seconds=DEFAULT_SHADOW_HOURS * 3600,
        )
        logger.info("Canary traffic bumped to %.0f%% for %s", new_pct * 100, candidate_version)
        return {"action": "ramp", "traffic_pct": new_pct, "improvement": improvement}

    if improvement < -mae_threshold:
        return await rollback_canary(
            store=store, cache=cache, kafka=kafka,
            candidate_version=candidate_version, baseline_version=baseline_version,
            segment=segment, region=region,
            reason=f"regression: candidate MAE {c_mae:.0f} vs baseline {b_mae:.0f}",
        )

    return {"action": "hold", "reason": "within_threshold", "improvement": improvement}


async def promote_model(
    *,
    store: PostgresStore,
    cache: RedisCache,
    kafka: KafkaBus | None = None,
    version: str,
    segment: str,
    region: str,
    shadow_mae: float | None = None,
) -> dict[str, Any]:
    await store.update_model_promotion(version, promotion_status="promoted", shadow_mae=shadow_mae)
    await store.set_active_model(version, segment, region)
    await cache.set_json(f"active_model:{segment}:{region}", {"version": version}, ttl_seconds=86400 * 30)

    canaries = await store.get_active_canaries()
    for c in canaries:
        if c["candidate_version"] == version:
            await store.upsert_canary_state({**c, "status": "promoted", "promoted_at": datetime.now(timezone.utc), "traffic_pct": 1.0})

    event = {"action": "promoted", "version": version, "segment": segment, "region": region, "shadow_mae": shadow_mae}
    if kafka is not None:
        await kafka.publish(CANARY_ALERT_TOPIC, event, key=version)
    logger.info("Model promoted: %s for %s/%s (MAE=%.1f)", version, region, segment, shadow_mae or 0)
    return event


async def rollback_canary(
    *,
    store: PostgresStore,
    cache: RedisCache,
    kafka: KafkaBus | None = None,
    candidate_version: str,
    baseline_version: str,
    segment: str,
    region: str,
    reason: str = "regression",
) -> dict[str, Any]:
    await store.update_model_promotion(candidate_version, promotion_status="rolled_back")

    canaries = await store.get_active_canaries()
    for c in canaries:
        if c["candidate_version"] == candidate_version:
            await store.upsert_canary_state({**c, "status": "rolled_back"})

    await cache.set_json(
        f"canary:{segment}:{region}",
        {"candidate": None, "baseline": baseline_version, "traffic_pct": 0.0},
        ttl_seconds=3600,
    )

    event = {"action": "rolled_back", "candidate": candidate_version, "baseline": baseline_version, "segment": segment, "region": region, "reason": reason}
    if kafka is not None:
        await kafka.publish(CANARY_ALERT_TOPIC, event, key=candidate_version)
    logger.warning("Canary rolled back: %s — %s", candidate_version, reason)
    return event
