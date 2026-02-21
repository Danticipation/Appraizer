from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


def mae(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    arr_true = np.asarray(y_true, dtype=float)
    arr_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(arr_true - arr_pred)))


def rmse(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    arr_true = np.asarray(y_true, dtype=float)
    arr_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((arr_true - arr_pred) ** 2)))


@dataclass
class CanaryController:
    traffic_percent: float = 0.10
    promote_threshold_mae_delta: float = 50.0
    max_percent: float = 1.0

    def evaluate_and_promote(
        self,
        y_true: Sequence[float],
        baseline_preds: Sequence[float],
        candidate_preds: Sequence[float],
        step: float = 0.10,
    ) -> bool:
        baseline_mae = mae(y_true, baseline_preds)
        candidate_mae = mae(y_true, candidate_preds)
        delta = baseline_mae - candidate_mae
        if delta >= self.promote_threshold_mae_delta:
            self.traffic_percent = min(self.max_percent, self.traffic_percent + step)
            return True
        return False

