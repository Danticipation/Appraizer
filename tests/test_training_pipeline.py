import numpy as np
import pandas as pd

from appraisal.config import TrainingConfig
from appraisal.training_pipeline import train_segmented_models


def _synthetic_data(n: int = 650) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    regions = np.array(["US-CA", "US-TX"])
    segments = np.array(["sedan", "suv"])
    title = np.array(["clean", "clean", "clean", "salvage", "rebuilt"])
    rows = []
    for i in range(n):
        year = int(rng.integers(2008, 2025))
        mileage = int(rng.integers(5_000, 180_000))
        t = str(rng.choice(title))
        rare = bool(rng.random() < 0.06)
        base = 28000 - 1200 * (2026 - year) - 0.045 * mileage
        if t in {"salvage", "rebuilt"}:
            base *= 0.65
        if rare:
            base *= 0.90
        noise = rng.normal(0, 1200)
        rows.append(
            {
                "vin": f"VIN{i:014d}"[:17],
                "region": str(rng.choice(regions)),
                "segment": str(rng.choice(segments)),
                "title_status": t,
                "has_rare_modification": rare,
                "mileage": mileage,
                "year": year,
                "image_damage_score": float(rng.random()),
                "image_tamper_score": float(rng.random() * 0.5),
                "obd_health_score": float(0.4 + rng.random() * 0.6),
                "obd_weighted_dtc_severity": float(rng.random()),
                "auction_close_price": float(max(700.0, base + noise)),
                "observed_at_epoch": int(1_700_000_000 + i * 86400),
            }
        )
    return pd.DataFrame(rows)


def test_segmented_training_pipeline_produces_models():
    cfg = TrainingConfig(min_samples_per_model=80)
    df = _synthetic_data()
    models = train_segmented_models(df, cfg)
    assert models
    any_model = next(iter(models.values()))
    assert any_model.mae >= 0
    assert any_model.rmse >= 0

