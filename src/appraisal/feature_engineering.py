from __future__ import annotations

from typing import Tuple

import pandas as pd


def infer_population(title_status: str, has_rare_modification: bool) -> str:
    t = title_status.lower()
    if t in {"salvage", "rebuilt"}:
        return "salvage_rebuilt"
    if has_rare_modification:
        return "rare_modified"
    return "normal"


def build_features(df: pd.DataFrame, reference_year: int = 2026) -> Tuple[pd.DataFrame, pd.Series]:
    frame = df.copy()
    frame["vehicle_age"] = (reference_year - frame["year"]).clip(lower=0)
    frame["population"] = [
        infer_population(t, m)
        for t, m in zip(frame["title_status"], frame["has_rare_modification"], strict=False)
    ]
    y = frame["auction_close_price"].astype(float)
    return frame, y

