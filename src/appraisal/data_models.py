from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


VehiclePopulation = Literal["normal", "salvage_rebuilt", "rare_modified"]


@dataclass(frozen=True)
class AppraisalRecord:
    vin: str
    region: str
    segment: str
    title_status: str
    has_rare_modification: bool
    mileage: int
    year: int
    image_damage_score: float
    image_tamper_score: float
    obd_health_score: float
    obd_weighted_dtc_severity: float
    auction_close_price: float
    observed_at_epoch: int

