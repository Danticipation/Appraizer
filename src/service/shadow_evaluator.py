from __future__ import annotations

import logging
import random
from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np
import pandas as pd

from service.messaging import KafkaBus
from service.storage import PostgresStore
from service.valuations import ValuationsFacade

logger = logging.getLogger(__name__)

SHADOW_ALERT_TOPIC = "shadow_alerts"

MAE_ALERT_THRESHOLD = 1500.0
BIAS_ALERT_THRESHOLD = 800.0
CALIBRATION_ALERT_THRESHOLD = 0.15


def compute_segment_metrics(df: pd.DataFrame) -> dict[str, Any]:
    predicted = df["predicted_value"].to_numpy(dtype=float)
    actual = df["actual_close_price"].to_numpy(dtype=float)
    errors = predicted - actual
    abs_errors = np.abs(errors)

    mae = float(np.mean(abs_errors))
    rmse = float(np.sqrt(np.mean(errors ** 2)))
    bias = float(np.mean(errors))

    total = len(df)
    non_rejected = int((~df["route_to_manual"].astype(bool)).sum())
    coverage_pct = non_rejected / max(total, 1) * 100.0

    confidence = df["confidence"].to_numpy(dtype=float)
    high_conf_mask = confidence >= 0.90
    if high_conf_mask.sum() > 0:
        high_conf_error_rate = float(np.mean(abs_errors[high_conf_mask] > mae))
        calibration_error = abs(1.0 - 0.90 - high_conf_error_rate)
    else:
        calibration_error = 0.0

    return {
        "mae": round(mae, 2),
        "rmse": round(rmse, 2),
        "bias": round(bias, 2),
        "coverage_pct": round(coverage_pct, 2),
        "calibration_error": round(calibration_error, 4),
        "sample_count": total,
    }


def generate_mock_actuals(df: pd.DataFrame, noise_pct: float = 0.10) -> pd.DataFrame:
    """Simulate realized auction closes as predicted +/- random offset for prototyping."""
    rng = random.Random(42)
    actuals = []
    for _, row in df.iterrows():
        offset = rng.uniform(-noise_pct, noise_pct) * row["predicted_value"]
        actuals.append(round(row["predicted_value"] + offset, 2))
    result = df.copy()
    result["actual_close_price"] = actuals
    return result


async def ingest_actuals_from_feed(
    *,
    store: PostgresStore,
    valuations: ValuationsFacade | None,
    window_days: int = 14,
) -> dict[str, Any]:
    """Pull realized prices for recent appraisals from MMR/third-party and store in feedback_actuals."""
    if valuations is None:
        return {"status": "skipped", "reason": "no_valuations_client"}

    recent = await store.get_recent_appraisals(limit=500)
    if not recent:
        return {"status": "skipped", "reason": "no_appraisals"}

    cutoff = datetime.now(timezone.utc) - timedelta(days=window_days)
    eligible = [
        r for r in recent
        if r.get("created_at") and r["created_at"] <= cutoff
    ]
    if not eligible:
        return {"status": "skipped", "reason": "no_settled_appraisals"}

    vins = list({r["vin"] for r in eligible})
    results = await valuations.get_actual_prices_batch(vins)

    ingested = 0
    for vr in results:
        if vr.wholesale_value is None:
            continue
        matching = [r for r in eligible if r["vin"] == vr.vin]
        if not matching:
            continue
        ref = matching[0]
        await store.insert_feedback_actual(
            vin=vr.vin,
            region=ref["region"],
            segment=ref["segment"],
            actual_close_price=vr.wholesale_value,
            source_json={"source": vr.source, "raw": vr.raw},
        )
        ingested += 1

    return {"status": "ok", "vins_queried": len(vins), "actuals_ingested": ingested}


async def run_shadow_evaluation(
    *,
    store: PostgresStore,
    kafka: KafkaBus | None = None,
    valuations: ValuationsFacade | None = None,
    window_days: int = 14,
    mae_threshold: float = MAE_ALERT_THRESHOLD,
    bias_threshold: float = BIAS_ALERT_THRESHOLD,
    calibration_threshold: float = CALIBRATION_ALERT_THRESHOLD,
    use_mock_actuals: bool = False,
) -> dict[str, Any]:
    now = datetime.now(timezone.utc)
    window_start = now - timedelta(days=window_days)
    window_end = now

    if valuations is not None:
        ingestion = await ingest_actuals_from_feed(store=store, valuations=valuations, window_days=window_days)
        logger.info("Actuals ingestion: %s", ingestion)

    df = await store.fetch_appraisals_with_actuals(window_start, window_end)

    if df.empty and use_mock_actuals:
        recent = await store.get_recent_appraisals(limit=200)
        if recent:
            df = pd.DataFrame(recent)
            df = generate_mock_actuals(df)

    if df.empty:
        logger.info("Shadow evaluation: no matched appraisals found in window.")
        return {"status": "skipped", "reason": "no_data"}

    results: list[dict[str, Any]] = []
    alerts: list[dict[str, Any]] = []

    groups = df.groupby(["model_version", "segment", "region", "population_segment"], dropna=False)
    for (model_version, segment, region, pop_seg), group_df in groups:
        if len(group_df) < 5:
            continue

        metrics = compute_segment_metrics(group_df)
        record = {
            "model_version": str(model_version),
            "segment": str(segment),
            "region": str(region),
            "population_segment": str(pop_seg),
            "window_start": window_start,
            "window_end": window_end,
            **metrics,
        }
        await store.insert_shadow_metric(record)
        results.append(record)

        degraded = False
        reasons = []
        if metrics["mae"] > mae_threshold:
            degraded = True
            reasons.append(f"MAE {metrics['mae']:.0f} > {mae_threshold:.0f}")
        if abs(metrics["bias"]) > bias_threshold:
            degraded = True
            reasons.append(f"|bias| {abs(metrics['bias']):.0f} > {bias_threshold:.0f}")
        if metrics["calibration_error"] > calibration_threshold:
            degraded = True
            reasons.append(f"calibration {metrics['calibration_error']:.4f} > {calibration_threshold}")

        if degraded:
            alert = {
                "model_version": str(model_version),
                "segment": str(segment),
                "region": str(region),
                "reasons": reasons,
                "metrics": metrics,
                "timestamp": now.isoformat(),
            }
            alerts.append(alert)
            logger.warning("Shadow degradation: %s/%s/%s — %s", model_version, region, segment, ", ".join(reasons))
            if kafka is not None:
                await kafka.publish(SHADOW_ALERT_TOPIC, alert, key=str(model_version))

    return {
        "status": "ok",
        "evaluated_groups": len(results),
        "alerts_fired": len(alerts),
        "results": results,
        "alerts": alerts,
    }
