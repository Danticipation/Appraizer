import pandas as pd

from appraisal.population import ks_distribution_mismatch, split_populations
from appraisal.routing import ManualReviewRouter
from appraisal.shadow_canary import CanaryController


def test_manual_review_router():
    router = ManualReviewRouter(confidence_threshold=0.95)
    reject = router.route({"vin": "x"}, confidence=0.80)
    assert reject is True
    assert router.queue.qsize() == 1


def test_canary_promotion():
    controller = CanaryController(traffic_percent=0.10, promote_threshold_mae_delta=10.0)
    y_true = [10000, 12000, 13000]
    baseline = [11000, 13000, 14000]
    candidate = [10050, 11950, 13100]
    promoted = controller.evaluate_and_promote(y_true, baseline, candidate, step=0.10)
    assert promoted is True
    assert controller.traffic_percent > 0.10


def test_population_split_and_ks():
    df = pd.DataFrame(
        [
            {"title_status": "clean", "has_rare_modification": False, "mileage": 40000},
            {"title_status": "salvage", "has_rare_modification": False, "mileage": 50000},
            {"title_status": "clean", "has_rare_modification": True, "mileage": 35000},
        ]
    )
    parts = split_populations(df)
    assert len(parts["normal"]) == 1
    assert len(parts["salvage_rebuilt"]) == 1
    assert len(parts["rare_modified"]) == 1

    report = ks_distribution_mismatch(
        pd.Series([1, 2, 3, 4, 5]),
        pd.Series([1, 2, 3, 4, 20]),
        alpha=0.05,
    )
    assert 0 <= report.p_value <= 1

