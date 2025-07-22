"""ID and trail styling utilities."""

from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd

from FIT_python.data_split_and_summary.data_import_utils import sanitize_labels

BASE_DIR = Path(__file__).resolve().parents[2] / "results" / "experiments" / "fit_start_to_finish"
CSV_GLOB = "*_baseline_predictions.csv"

FILLED_MARKERS = {"o", "s", "^", "v", "P", "X", "D", "*", "h", "8"}
_BASE_MARKERS = [
    "o",
    "s",
    "^",
    "v",
    "P",
    "X",
    "D",
    "*",
    "h",
    "+",
    "x",
    "1",
    "2",
    "3",
    "4",
    "8",
]


def sanitize_id_trail(df: pd.DataFrame, id_col: str = "individual_id", trail_col: str = "trail") -> pd.DataFrame:
    """Return ``df`` with cleaned ``id_col`` and ``trail_col`` values."""

    cols = [c for c in (id_col, trail_col) if c in df.columns]
    if not cols:
        return df.copy()
    return sanitize_labels(df, cols)


def _load_unique() -> tuple[list[str], list[str]]:
    ids: set[str] = set()
    trails: set[str] = set()
    for fp in BASE_DIR.glob(CSV_GLOB):
        usecols = ["individual_id", "trail"]
        df = pd.read_csv(fp, usecols=lambda c: c in usecols)
        ids.update(df["individual_id"].dropna().astype(str))
        if "trail" in df.columns:
            trails.update(df["trail"].dropna().astype(str))
    ids_df = pd.DataFrame({"individual_id": list(ids)})
    trails_df = pd.DataFrame({"trail": list(trails)})
    ids_clean = sanitize_labels(ids_df, ["individual_id"])["individual_id"].sort_values().unique().tolist()
    trails_clean = sanitize_labels(trails_df, ["trail"])["trail"].sort_values().unique().tolist()
    return ids_clean, trails_clean


_IDS, _TRAILS = _load_unique()

_COLORS = list(map(mpl.colors.to_hex, plt.cm.tab20.colors))


def _build_palette(labels: list[str]) -> dict[str, str]:
    return {lab: _COLORS[i % len(_COLORS)] for i, lab in enumerate(labels)}


def _build_markers(labels: list[str]) -> dict[str, str]:
    from itertools import cycle

    return {lab: m for lab, m in zip(labels, cycle(_BASE_MARKERS))}


ID_COLORS = _build_palette(_IDS)
ID_MARKERS = _build_markers(_IDS)
TRAIL_COLORS = _build_palette(_TRAILS)
TRAIL_MARKERS = _build_markers(_TRAILS)

__all__ = [
    "sanitize_id_trail",
    "ID_COLORS",
    "ID_MARKERS",
    "TRAIL_COLORS",
    "TRAIL_MARKERS",
]
