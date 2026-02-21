from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from service.messaging import KafkaBus

logger = logging.getLogger(__name__)

DRIFT_ALERT_TOPIC = "drift_alerts"

KS_P_VALUE_THRESHOLD = 0.01
PSI_THRESHOLD = 0.20
PSI_BINS = 10


def population_stability_index(
    reference: np.ndarray, current: np.ndarray, bins: int = PSI_BINS,
) -> float:
    eps = 1e-6
    combined = np.concatenate([reference, current])
    breakpoints = np.linspace(np.min(combined), np.max(combined) + eps, bins + 1)

    ref_counts = np.histogram(reference, bins=breakpoints)[0].astype(float)
    cur_counts = np.histogram(current, bins=breakpoints)[0].astype(float)

    ref_pct = ref_counts / max(ref_counts.sum(), 1) + eps
    cur_pct = cur_counts / max(cur_counts.sum(), 1) + eps

    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def run_drift_checks(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    feature_cols: list[str] | None = None,
    ks_threshold: float = KS_P_VALUE_THRESHOLD,
    psi_threshold: float = PSI_THRESHOLD,
) -> dict[str, Any]:
    if feature_cols is None:
        feature_cols = ["mileage", "year", "image_damage_score", "obd_health_score"]

    drifted_features: list[dict[str, Any]] = []
    all_results: dict[str, dict[str, Any]] = {}

    for col in feature_cols:
        if col not in reference_df.columns or col not in current_df.columns:
            continue

        ref_vals = reference_df[col].dropna().to_numpy(dtype=float)
        cur_vals = current_df[col].dropna().to_numpy(dtype=float)

        if len(ref_vals) < 10 or len(cur_vals) < 10:
            continue

        ks_stat, ks_p = stats.ks_2samp(ref_vals, cur_vals)
        psi = population_stability_index(ref_vals, cur_vals)

        result = {
            "ks_statistic": round(float(ks_stat), 4),
            "ks_p_value": round(float(ks_p), 6),
            "psi": round(psi, 4),
            "ks_drifted": ks_p < ks_threshold,
            "psi_drifted": psi > psi_threshold,
            "ref_mean": round(float(np.mean(ref_vals)), 2),
            "cur_mean": round(float(np.mean(cur_vals)), 2),
            "ref_std": round(float(np.std(ref_vals)), 2),
            "cur_std": round(float(np.std(cur_vals)), 2),
        }
        all_results[col] = result

        if result["ks_drifted"] or result["psi_drifted"]:
            drifted_features.append({"feature": col, **result})

    return {
        "drift_detected": len(drifted_features) > 0,
        "drifted_features": drifted_features,
        "all_results": all_results,
        "features_checked": len(all_results),
    }


async def run_drift_check_and_alert(
    *,
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    model_version: str,
    kafka: KafkaBus | None = None,
    feature_cols: list[str] | None = None,
) -> dict[str, Any]:
    result = run_drift_checks(reference_df, current_df, feature_cols=feature_cols)

    if result["drift_detected"]:
        alert = {
            "model_version": model_version,
            "drifted_features": result["drifted_features"],
            "features_checked": result["features_checked"],
        }
        logger.warning(
            "Drift detected for %s: %s",
            model_version,
            [f["feature"] for f in result["drifted_features"]],
        )
        if kafka is not None:
            await kafka.publish(DRIFT_ALERT_TOPIC, alert, key=model_version)
        result["alert_published"] = True
    else:
        logger.info("No drift detected for %s (%d features checked)", model_version, result["features_checked"])
        result["alert_published"] = False

    return result
