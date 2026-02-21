from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from appraisal.shadow_canary import CanaryController, mae, rmse


@dataclass
class ShadowEvaluationResult:
    baseline_mae: float
    candidate_mae: float
    baseline_rmse: float
    candidate_rmse: float
    promoted: bool
    active_traffic_percent: float


def run_shadow_then_canary(
    y_true: Sequence[float],
    baseline_preds: Sequence[float],
    candidate_preds: Sequence[float],
    controller: CanaryController,
) -> ShadowEvaluationResult:
    promoted = controller.evaluate_and_promote(y_true, baseline_preds, candidate_preds)
    return ShadowEvaluationResult(
        baseline_mae=mae(y_true, baseline_preds),
        candidate_mae=mae(y_true, candidate_preds),
        baseline_rmse=rmse(y_true, baseline_preds),
        candidate_rmse=rmse(y_true, candidate_preds),
        promoted=promoted,
        active_traffic_percent=controller.traffic_percent,
    )

