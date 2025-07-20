from __future__ import annotations

"""Utility helpers for a simple ID baseline."""

from pathlib import Path
from typing import List, Dict, Any

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from FIT_python.caption_utils import save_caption
from FIT_python.Visualisations.plot_style import apply_style
from .population_estimation import concordance_correlation_coefficient


def collect_id_metrics(exp_dir: Path) -> pd.DataFrame:
    """Return averaged metrics across baseline holdout splits.

    Parameters
    ----------
    exp_dir:
        Root directory containing one subdirectory per species with a
        ``summary.csv`` produced by the baseline helper.

    Returns
    -------
    pandas.DataFrame
        Table with one row per species containing the mean BCR, mean ERD
        and the concordance correlation coefficient (CCC).
    """
    exp_dir = Path(exp_dir)
    csv_files = sorted(exp_dir.glob("*/summary.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No 'summary.csv' found under {exp_dir}")

    records: List[Dict[str, Any]] = []
    for csv in csv_files:
        df = pd.read_csv(csv)
        if df.empty:
            continue
        mean_bcr = df["bcr"].mean() if "bcr" in df.columns else float("nan")
        mean_erd = df["erd"].mean() if "erd" in df.columns else float("nan")
        if "ccc" in df.columns and not df["ccc"].isna().all():
            ccc = float(df["ccc"].iloc[0])
        elif {"pred_count", "true_count"}.issubset(df.columns):
            ccc = concordance_correlation_coefficient(
                df["pred_count"], df["true_count"]
            )
        else:
            ccc = float("nan")
        records.append({"species": csv.parent.name, "bcr": mean_bcr, "erd": mean_erd, "ccc": ccc})

    result = pd.DataFrame(records)
    result.to_csv(exp_dir / "raw_results.csv", index=False)
    return result


def plot_bcr_comparison(df: pd.DataFrame, fig_dir: Path) -> Path:
    """Plot a bar chart comparing BCR across species."""
    if "species" not in df.columns or "bcr" not in df.columns:
        raise KeyError("DataFrame must contain 'species' and 'bcr'")

    apply_style()
    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    order = df.sort_values("bcr", ascending=False)["species"]

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(data=df, x="species", y="bcr", order=order, ax=ax, color="#4C72B0", edgecolor="black")
    ax.set_xlabel("Species")
    ax.set_ylabel("BCR")
    plt.xticks(rotation=45, ha="right")
    fig.tight_layout()

    out = fig_dir / "bcr_comparison.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    save_caption(out, "Baseline BCR per species")
    return out
