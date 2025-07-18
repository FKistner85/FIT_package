from __future__ import annotations

"""Utility helpers for the individual ID pipelines."""

from typing import Iterable, List
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


