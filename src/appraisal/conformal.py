from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd

try:
    from mapie.regression import MapieRegressor
except ModuleNotFoundError:  # pragma: no cover
    MapieRegressor = None

class Predictor(Protocol):
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None: ...
    def predict(self, X: pd.DataFrame) -> np.ndarray: ...


@dataclass
class ConformalRegressor:
    model: Predictor
    alpha: float = 0.05
    q_hat: float = 0.0

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series, X_cal: pd.DataFrame, y_cal: pd.Series) -> None:
        self.model.fit(X_train, y_train)
        cal_preds = self.model.predict(X_cal)
        residuals = np.abs(y_cal.to_numpy() - cal_preds)
        self.q_hat = float(np.quantile(residuals, 1 - self.alpha))

    def predict_with_interval(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        preds = self.model.predict(X)
        lower = preds - self.q_hat
        upper = preds + self.q_hat
        return preds, lower, upper


@dataclass
class MapieConformalRegressor:
    """
    Optional MAPIE-based conformal wrapper for sklearn-compatible estimators.
    """

    estimator: object
    alpha: float = 0.05

    def __post_init__(self) -> None:
        if MapieRegressor is None:
            raise RuntimeError("mapie is not installed")
        self.mapie = MapieRegressor(estimator=self.estimator, method="plus")

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        self.mapie.fit(X_train, y_train)

    def predict_with_interval(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        preds, interval = self.mapie.predict(X, alpha=self.alpha)
        lower = interval[:, 0, 0]
        upper = interval[:, 1, 0]
        return preds, lower, upper

