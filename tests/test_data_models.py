from appraisal.data_models import AppraisalRecord


def test_appraisal_record_shape():
    rec = AppraisalRecord(
        vin="1HGCM82633A123456",
        region="US-CA",
        segment="sedan",
        title_status="clean",
        has_rare_modification=False,
        mileage=100000,
        year=2017,
        image_damage_score=0.12,
        image_tamper_score=0.08,
        obd_health_score=0.90,
        obd_weighted_dtc_severity=0.10,
        auction_close_price=12850.0,
        observed_at_epoch=1700000000,
    )
    assert rec.vin.endswith("3456")

