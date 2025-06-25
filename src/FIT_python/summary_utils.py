# src/FIT_python/summary_utils.py
"""Utility functions for dataset split summaries."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


# ---------------------------------------------------------------------------
# Discovery & Loading
# ---------------------------------------------------------------------------

def discover_splits(splits_dir: Path) -> Dict[str, Dict[str, Path]]:
    """Scan ``splits_dir`` for split files.

    Returns mapping ``dataset -> {origin -> path}``.
    Handles files either in ``<dataset>_<origin>.parquet`` format
    or ``<dataset>/<origin>.parquet`` layout.
    """
    results: Dict[str, Dict[str, Path]] = {}
    for path in sorted(splits_dir.rglob("*.parquet")):
        if path.parent == splits_dir:
            if "_" not in path.stem:
                # Skip files not matching pattern
                continue
            dataset, origin = path.stem.rsplit("_", 1)
        else:
            dataset = path.parent.name
            origin = path.stem
        dataset = dataset.strip().replace(" ", "_").lower()
        origin = origin.strip().lower()
        results.setdefault(dataset, {})[origin] = path
    return results


def load_split_data(path: Path) -> pd.DataFrame:
    """Load a single split file and validate required columns."""
    if not path.exists():
        raise RuntimeError(f"Split file not found: {path}")
    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    elif path.suffix == ".csv":
        df = pd.read_csv(path)
    else:
        raise RuntimeError(f"Unsupported file type: {path.suffix}")
    if "sex" not in df.columns:
        raise RuntimeError(f"Missing 'sex' column in {path}")
    return df


# ---------------------------------------------------------------------------
# Summary & Plotting
# ---------------------------------------------------------------------------

def compute_summary(df: pd.DataFrame, dataset: str, origin: str) -> pd.DataFrame:
    """Return table with counts per sex for one dataset/origin."""
    sex_series = df["sex"].fillna("Unknown").astype(str).str.strip()
    counts = sex_series.value_counts()
    rows = []
    is_otter = "otter" in dataset.lower() or origin.lower() == "inference"
    for sex in ["F", "M", "Unknown"]:
        rows.append(
            {
                "Dataset": dataset,
                "Origin": origin.capitalize(),
                "Sex": sex,
                "Count": int(counts.get(sex, 0)),
                "IsOtter": is_otter,
            }
        )
    return pd.DataFrame(rows)


def _lighten(color: str, amount: float) -> str:
    base = mcolors.to_rgb(color)
    r, g, b = [1 - (1 - c) * amount for c in base]
    return mcolors.to_hex((r, g, b))


def plot_summary_table(df_summary: pd.DataFrame, fig_dir: Path) -> None:
    """Create a bar plot for each dataset summarising sex counts."""
    fig_dir.mkdir(parents=True, exist_ok=True)

    base_colors = {"F": "#d62728", "M": "#1f77b4", "Unknown": "#888888"}
    shade = {"train": 0.6, "test": 1.0, "inference": 0.8}

    for dataset in sorted(df_summary["Dataset"].unique()):
        sub = df_summary[df_summary["Dataset"] == dataset]
        origins = list(sub["Origin"].str.lower().unique())
        sexes = ["F", "M", "Unknown"]
        width = 0.8 / max(len(origins), 1)

        fig, ax = plt.subplots(figsize=(6, 4))
        for i, origin in enumerate(origins):
            df_o = sub[sub["Origin"].str.lower() == origin]
            counts = [int(df_o[df_o["Sex"] == s]["Count"].sum()) for s in sexes]
            pos = [x + (i - len(origins)/2) * width for x in range(len(sexes))]
            cols = [
                _lighten(base_colors[s], shade.get(origin, 1.0)) for s in sexes
            ]
            hatch = "//" if origin == "inference" else None
            ax.bar(pos, counts, width=width, color=cols, edgecolor="black", hatch=hatch, label=origin.capitalize())

        ax.set_xticks(range(len(sexes)))
        ax.set_xticklabels(sexes)
        ax.set_ylabel("Count")
        title = dataset
        if sub["IsOtter"].any():
            title += " [Otter]"
        ax.set_title(title)
        ax.legend(title="Origin")

        fig.tight_layout()
        out_path = fig_dir / f"{dataset}_summary.png"
        fig.savefig(out_path)
        plt.close(fig)

