from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple
import pandas as pd


def map_indices(
    id_list: Iterable[str | int],
    idx_to_id: pd.Series,
    valid_index: Iterable[str],
) -> List[str]:
    """Map positional indices to ID strings if necessary.

    Parameters
    ----------
    id_list:
        List of ID strings or positional indices.
    idx_to_id:
        Series mapping positional indices to ID strings.
    valid_index:
        Iterable of valid ID strings (e.g., ``DataFrame.index``).

    Returns
    -------
    List[str]
        The mapped ID strings.
    """

    mapped: List[str] = []
    for x in id_list:
        sx = str(x)
        if sx in valid_index:
            mapped.append(sx)
            continue
        try:
            idx = int(x)
        except (ValueError, TypeError):
            raise KeyError(f"ID '{x}' not found in DataFrame")
        if idx < 0 or idx >= len(idx_to_id):
            raise KeyError(f"ID '{x}' not found in DataFrame")
        mapped.append(idx_to_id.iloc[idx])
    return mapped


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



