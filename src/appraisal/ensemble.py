from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from appraisal.config import TrainingConfig
from appraisal.model_components import DeepTabularModel, GBMModel


@dataclass
class RuleLayer:
    """
    Deterministic rule layer applied after stacked model output.
    """

    def adjust(self, raw_preds: np.ndarray, frame: pd.DataFrame) -> np.ndarray:
        adjusted = raw_preds.astype(float).copy()
        title = frame["title_status"].str.lower()
        salvage_mask = title.eq("salvage")
        rebuilt_mask = title.eq("rebuilt")
        rare_mod_mask = frame["has_rare_modification"].astype(bool)
        tamper_mask = frame["image_tamper_score"] > 0.65

        adjusted[salvage_mask.values] *= 0.58
        adjusted[rebuilt_mask.values] *= 0.72
        adjusted[rare_mod_mask.values] *= 0.90
        adjusted[tamper_mask.values] *= 0.92
        return np.clip(adjusted, 300.0, None)


class StackedEnsemble:
    def __init__(self, config: TrainingConfig) -> None:
        self.config = config
        self.gbm = GBMModel(config.categorical_cols, config.numeric_cols)
        self.deep = DeepTabularModel(config.categorical_cols, config.numeric_cols)
        self.meta = Ridge(alpha=1.0)
        self.rules = RuleLayer()
        self.is_fitted = False

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        self.gbm.fit(X, y)
        self.deep.fit(X, y)
        gbm_preds = self.gbm.predict(X)
        deep_preds = self.deep.predict(X)
        stacked = np.column_stack([gbm_preds, deep_preds])
        self.meta.fit(stacked, y.to_numpy())
        self.is_fitted = True

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("StackedEnsemble not fitted")
        gbm_preds = self.gbm.predict(X)
        deep_preds = self.deep.predict(X)
        stacked = np.column_stack([gbm_preds, deep_preds])
        raw = self.meta.predict(stacked)
        return self.rules.adjust(raw, X)


def split_by_region_segment(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    groups: Dict[str, pd.DataFrame] = {}
    for (region, segment), frame in df.groupby(["region", "segment"], sort=False):
        groups[f"{region}::{segment}"] = frame
    return groups

