"""Helpers for anonymising individual and trail labels."""

from __future__ import annotations

import pandas as pd


def generate_display_mapping(
    df: pd.DataFrame,
    id_col: str = "individual_id",
    trail_col: str = "trail",
) -> dict[str, str]:
    """Return mapping from original labels to anonymised display labels."""
    mapping: dict[str, str] = {}
    if id_col not in df.columns:
        return mapping
    ids = df[id_col].dropna().astype(str).unique()
    for i, orig_id in enumerate(ids, start=1):
        disp_id = f"Ind_{i}"
        mapping[str(orig_id)] = disp_id
        if trail_col in df.columns:
            trails = (
                df.loc[df[id_col] == orig_id, trail_col]
                .dropna()
                .astype(str)
                .unique()
            )
            for j, tr in enumerate(trails, start=1):
                mapping[str(tr)] = f"{disp_id}_trail_orig_{j}"
    return mapping


def apply_display_mapping(
    df: pd.DataFrame,
    mapping: dict[str, str],
    id_col: str = "individual_id",
    trail_col: str = "trail",
) -> pd.DataFrame:
    """Return ``df`` with ``display_id`` and ``display_trail`` columns added."""
    out = df.copy()
    if id_col in out.columns:
        out["display_id"] = (
            out[id_col].astype(str).map(mapping).fillna(out[id_col].astype(str))
        )
    if trail_col in out.columns:
        out["display_trail"] = (
            out[trail_col]
            .astype(str)
            .map(mapping)
            .fillna(out[trail_col].astype(str))
        )
    return out
