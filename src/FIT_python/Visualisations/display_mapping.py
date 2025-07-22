from __future__ import annotations

from typing import Dict
import pandas as pd


def generate_display_mapping(
    df: pd.DataFrame, id_col: str = "individual_id", trail_col: str = "trail"
) -> dict:
    """Return anonymised mapping for IDs and trails.

    Each unique ``id_col`` value is mapped to ``"Ind_<n>"``. For trails the
    mapping is ``"<display_id>_trail_orig_<n>"`` with numbering per
    individual.
    """
    mapping: Dict[str, Dict[str, str]] = {"id": {}, "trail": {}}
    if id_col in df.columns:
        ids = sorted(df[id_col].dropna().astype(str).unique())
        mapping["id"] = {orig: f"Ind_{i+1}" for i, orig in enumerate(ids)}
    if trail_col in df.columns and trail_col in df:
        # build per-id numbering
        tr_df = df[[id_col, trail_col]].dropna(subset=[trail_col]).astype(str)
        for ind, grp in tr_df.groupby(id_col):
            disp_id = mapping["id"].get(ind, ind)
            for j, trail in enumerate(sorted(grp[trail_col].unique()), start=1):
                mapping["trail"][trail] = f"{disp_id}_trail_orig_{j}"
    return mapping


def apply_display_mapping(
    df: pd.DataFrame,
    mapping: dict,
    id_col: str = "individual_id",
    trail_col: str = "trail",
) -> pd.DataFrame:
    """Return ``df`` with ``display_id`` and ``display_trail`` columns added."""
    out = df.copy()
    id_map = mapping.get("id", {})
    if id_col in out.columns:
        out["display_id"] = out[id_col].astype(str).map(id_map).fillna(out[id_col])
    if trail_col in out.columns:
        trail_map = mapping.get("trail", {})
        out["display_trail"] = (
            out[trail_col].astype(str).map(trail_map).fillna(out[trail_col])
        )
    return out
