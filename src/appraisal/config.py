from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict


@dataclass(frozen=True)
class TrainingConfig:
    conformal_alpha: float = 0.05
    low_confidence_threshold: float = 0.95
    canary_initial_percent: float = 0.10
    canary_promote_mae_delta_threshold: float = 50.0
    min_samples_per_model: int = 250
    retrain_cron: str = "0 2 * * 1"  # Weekly: Monday 02:00
    categorical_cols: tuple[str, ...] = ("region", "segment", "title_status")
    numeric_cols: tuple[str, ...] = (
        "mileage",
        "vehicle_age",
        "image_damage_score",
        "image_tamper_score",
        "obd_health_score",
        "obd_weighted_dtc_severity",
    )
    population_map: Dict[str, str] = field(
        default_factory=lambda: {
            "salvage": "salvage_rebuilt",
            "rebuilt": "salvage_rebuilt",
            "rare_mod_true": "rare_modified",
        }
    )

