from __future__ import annotations

"""Utility helpers for the individual ID pipelines."""

from typing import Sequence, Tuple

import pandas as pd


def prepare_base_df(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    *,
    sample_col: str = "id",
) -> Tuple[pd.DataFrame, pd.Series]:
    """Return cleaned DataFrame and index mapping.

    Parameters
    ----------
    df:
        Input data frame with sample rows.
    feature_cols:
        Names of numeric feature columns.
    sample_col:
        Column containing sample identifiers.

    Returns
    -------
    Tuple[pd.DataFrame, pd.Series]
        ``df`` indexed by ``sample_col`` and a positional index to id mapping.
    """

    df2 = df.copy()
    df2[sample_col] = df2[sample_col].astype(str)
    df2[feature_cols] = df2[feature_cols].apply(pd.to_numeric, errors="coerce")
    df_base = df2.set_index(sample_col)
    idx_to_id = df2[sample_col].astype(str).reset_index(drop=True)
    return df_base, idx_to_id


