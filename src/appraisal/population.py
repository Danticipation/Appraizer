from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import pandas as pd
from scipy.stats import ks_2samp


@dataclass(frozen=True)
class DriftReport:
    statistic: float
    p_value: float
    drift_detected: bool


def split_populations(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    output = {
        "normal": df[(~df["title_status"].str.lower().isin(["salvage", "rebuilt"])) & (~df["has_rare_modification"])],
        "salvage_rebuilt": df[df["title_status"].str.lower().isin(["salvage", "rebuilt"])],
        "rare_modified": df[df["has_rare_modification"]],
    }
    return output


def ks_distribution_mismatch(train_series: pd.Series, infer_series: pd.Series, alpha: float = 0.05) -> DriftReport:
    statistic, p_val = ks_2samp(train_series.to_numpy(), infer_series.to_numpy())
    return DriftReport(statistic=float(statistic), p_value=float(p_val), drift_detected=bool(p_val < alpha))

