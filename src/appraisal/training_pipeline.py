from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from appraisal.config import TrainingConfig
from appraisal.conformal import ConformalRegressor
from appraisal.ensemble import StackedEnsemble
from appraisal.feature_engineering import build_features
from appraisal.population import split_populations
from appraisal.temporal_cv import temporal_group_splits


@dataclass
class TrainedGroupModel:
    group_key: str
    conformal: ConformalRegressor
    mae: float
    rmse: float
    samples: int


def train_segmented_models(df: pd.DataFrame, config: TrainingConfig) -> Dict[str, TrainedGroupModel]:
    frame, _ = build_features(df)
    output: Dict[str, TrainedGroupModel] = {}
    for pop_name, pop_df in split_populations(frame).items():
        if pop_df.empty:
            continue
        for fold in temporal_group_splits(pop_df, n_splits=4):
            fold_train = pop_df.loc[fold.train_idx]
            fold_test = pop_df.loc[fold.test_idx]
            if len(fold_train) < config.min_samples_per_model:
                continue
            train_cut = int(len(fold_train) * 0.85)
            train_df = fold_train.iloc[:train_cut]
            cal_df = fold_train.iloc[train_cut:]
            if cal_df.empty:
                continue

            model = StackedEnsemble(config)
            conformal = ConformalRegressor(model=model, alpha=config.conformal_alpha)
            conformal.fit(
                train_df,
                train_df["auction_close_price"],
                cal_df,
                cal_df["auction_close_price"],
            )
            preds, _, _ = conformal.predict_with_interval(fold_test)
            y_true = fold_test["auction_close_price"].to_numpy()
            model_key = f"{pop_name}::{fold.group_key}"
            output[model_key] = TrainedGroupModel(
                group_key=model_key,
                conformal=conformal,
                mae=float(mean_absolute_error(y_true, preds)),
                rmse=float(np.sqrt(mean_squared_error(y_true, preds))),
                samples=len(fold_train),
            )
    return output

