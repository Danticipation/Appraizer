from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib

from appraisal.config import TrainingConfig
from appraisal.training_pipeline import train_segmented_models
from service.drift_detector import run_drift_checks
from service.storage import PostgresStore


async def run_retraining(
    *,
    store: PostgresStore,
    artifact_dir: str,
    config: TrainingConfig,
    previous_frame: Any | None = None,
) -> dict[str, Any]:
    frame = await store.fetch_training_frame()
    if frame.empty:
        return {"status": "skipped", "reason": "no_training_data"}

    trained = train_segmented_models(frame, config)
    if not trained:
        return {"status": "skipped", "reason": "insufficient_samples"}

    out_dir = Path(artifact_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    version = datetime.now(timezone.utc).strftime("model-%Y%m%d%H%M%S")

    drift_results: dict[str, Any] = {}
    if previous_frame is not None and not previous_frame.empty:
        drift_results = run_drift_checks(previous_frame, frame)

    for key, result in trained.items():
        file_path = out_dir / f"{key.replace('::', '__')}.joblib"
        payload = {"key": key, "model": result.conformal.model, "q_hat": result.conformal.q_hat, "version": version}
        joblib.dump(payload, file_path)
        parts = key.split("::")
        population = parts[0]
        region = parts[1] if len(parts) > 1 else "unknown"
        segment = parts[2] if len(parts) > 2 else "unknown"
        await store.insert_model_version(
            segment=segment,
            region=region,
            population_segment=population,
            version=version,
            metrics_json={"mae": result.mae, "rmse": result.rmse, "samples": result.samples},
        )
        if drift_results:
            await store.update_model_promotion(
                version,
                promotion_status="pending",
                drift_metrics_json=drift_results.get("all_results", {}),
            )

    return {
        "status": "ok",
        "version": version,
        "trained_models": len(trained),
        "drift": drift_results if drift_results else None,
    }
