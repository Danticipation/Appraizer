import pandas as pd

from appraisal.temporal_cv import temporal_group_splits


def _make_df(n: int = 60) -> pd.DataFrame:
    rows = []
    for i in range(n):
        rows.append(
            {
                "region": "US-CA",
                "segment": "sedan",
                "observed_at_epoch": 1_700_000_000 + i * 86400,
            }
        )
    return pd.DataFrame(rows)


def test_temporal_group_splits_ordering():
    df = _make_df()
    folds = list(temporal_group_splits(df, n_splits=4))
    assert folds
    for fold in folds:
        assert df.loc[fold.train_idx, "observed_at_epoch"].max() < df.loc[fold.test_idx, "observed_at_epoch"].min()

