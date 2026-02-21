from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from appraisal.feature_engineering import infer_population


@dataclass
class InferenceOutput:
    prediction: float
    interval_low: float
    interval_high: float
    confidence: float
    model_version: str
    population_segment: str


class InferenceFacade:
    def __init__(self, artifact_dir: str) -> None:
        self.artifact_dir = Path(artifact_dir)
        self._models: dict[str, Any] = {}
        self._loaded = False

    def _baseline(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        baseline = 15000.0 - 0.05 * frame["mileage"].to_numpy() - 220.0 * (2026 - frame["year"].to_numpy())
        preds = np.clip(baseline, 500.0, None)
        interval = np.maximum(preds * 0.12, 500.0)
        return preds, preds - interval, preds + interval

    def _load_models_if_needed(self) -> None:
        if self._loaded:
            return
        if not self.artifact_dir.exists():
            self._loaded = True
            return
        for file_path in self.artifact_dir.glob("*.joblib"):
            try:
                payload = joblib.load(file_path)
                key = str(payload.get("key") or file_path.stem)
                self._models[key] = payload
            except Exception:
                continue
        self._loaded = True

    def predict(
        self,
        *,
        frame: pd.DataFrame,
        region: str,
        segment: str,
        title_status: str,
        has_rare_modification: bool,
    ) -> InferenceOutput:
        self._load_models_if_needed()
        population = infer_population(title_status, has_rare_modification)
        model_key = f"{population}::{region}::{segment}"
        payload = self._models.get(model_key)

        if payload and payload.get("model") is not None:
            model = payload["model"]
            preds = model.predict(frame)
            q_hat = float(payload.get("q_hat", 500.0))
            low = preds - q_hat
            high = preds + q_hat
            version = str(payload.get("version", "model-v1"))
        else:
            preds, low, high = self._baseline(frame)
            version = "baseline-v1"

        prediction = float(preds[0])
        interval_low = float(low[0])
        interval_high = float(high[0])
        width = max(1.0, interval_high - interval_low)
        confidence = max(0.01, min(0.999, 1.0 - (width / max(5000.0, prediction))))

        return InferenceOutput(
            prediction=prediction,
            interval_low=interval_low,
            interval_high=interval_high,
            confidence=confidence,
            model_version=version,
            population_segment=population,
        )
