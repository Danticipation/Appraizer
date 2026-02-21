from appraisal.deployment_pipeline import run_shadow_then_canary
from appraisal.scheduler import build_weekly_retrain_scheduler
from appraisal.shadow_canary import CanaryController


def test_scheduler_builds_job():
    scheduler = build_weekly_retrain_scheduler("0 2 * * 1", lambda: None)
    jobs = scheduler.get_jobs()
    assert len(jobs) == 1
    assert jobs[0].id == "weekly_retrain"


def test_shadow_then_canary_pipeline():
    controller = CanaryController(traffic_percent=0.10, promote_threshold_mae_delta=20.0)
    result = run_shadow_then_canary(
        y_true=[10000, 12000, 14000],
        baseline_preds=[11000, 13300, 15000],
        candidate_preds=[10020, 11990, 14100],
        controller=controller,
    )
    assert result.promoted is True
    assert result.active_traffic_percent > 0.10

