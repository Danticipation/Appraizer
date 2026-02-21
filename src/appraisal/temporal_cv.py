from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit


@dataclass(frozen=True)
class Fold:
    train_idx: np.ndarray
    test_idx: np.ndarray
    group_key: str


def temporal_group_splits(
    df: pd.DataFrame,
    time_col: str = "observed_at_epoch",
    n_splits: int = 4,
) -> Iterator[Fold]:
    """
    Applies TimeSeriesSplit independently per (region, segment),
    guaranteeing no future leakage across folds.
    """
    for (region, segment), frame in df.groupby(["region", "segment"], sort=False):
        ordered = frame.sort_values(time_col)
        if len(ordered) < n_splits + 2:
            continue
        splitter = TimeSeriesSplit(n_splits=n_splits)
        local_idx = ordered.index.to_numpy()
        for train_local, test_local in splitter.split(ordered):
            yield Fold(
                train_idx=local_idx[train_local],
                test_idx=local_idx[test_local],
                group_key=f"{region}::{segment}",
            )

