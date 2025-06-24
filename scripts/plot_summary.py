#!/usr/bin/env python3
"""Plot stacked bar charts summarizing dataset splits.

This script reads ``results/data/gesamt/summary_datasets.csv`` and for each
species (including the overall ``Total``) creates a stacked bar chart showing
counts of unique individuals per split.  Each bar segment contains the text
``n=(individuals, trails)``.  Resulting plots are written to ``results/plots``.
"""
from __future__ import annotations

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

from FIT_python.config import RESULTS_DATA_DIR, FIGURES_DIR


DATA_PATH = RESULTS_DATA_DIR / "gesamt" / "summary_datasets.csv"
OUT_DIR = FIGURES_DIR
SPLITS = ["train", "val", "test"]
SEXES = ["F", "M", "Unknown"]

# Colors for each (split, sex) combination
COLORS = {
    "train": {"F": "#800000", "M": "#000080", "Unknown": "#aaaaaa"},
    "val":   {"F": "#cc6666", "M": "#6666cc", "Unknown": "#bbbbbb"},
    "test":  {"F": "#ff9999", "M": "#9999ff", "Unknown": "#cccccc"},
}


def load_summary(path: Path) -> pd.DataFrame:
    """Load summary CSV and return rows aggregated by species/split/sex."""
    df = pd.read_csv(path)
    # Normalise fields
    df["Species"] = df["Species"].str.strip()
    df["Split"] = df["Split"].str.strip().str.lower()
    df["Sex"] = df["Sex"].str.strip()

    # Only keep aggregated rows
    df = df[df["Dataorigin"] == "All"]
    df = df[df["Split"].isin(SPLITS)]
    return df


def summary_for_species(df: pd.DataFrame, species: str) -> dict[str, dict[str, tuple[int, int]]]:
    """Return mapping split->sex->(individuals, trails) for one species."""
    sub = df[df["Species"] == species]
    out: dict[str, dict[str, tuple[int, int]]] = {s: {} for s in SPLITS}
    for split in SPLITS:
        df_split = sub[sub["Split"] == split]
        for sex in SEXES:
            df_sex = df_split[df_split["Sex"] == sex]
            if df_sex.empty:
                out[split][sex] = (0, 0)
            else:
                row = df_sex.iloc[0]
                out[split][sex] = (
                    int(row["UniqueIndividuals"]),
                    int(row["Trails"]),
                )
    return out


def plot_species(species: str, data: dict[str, dict[str, tuple[int, int]]]) -> None:
    """Create and save plot for one species."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4))

    x = range(len(SPLITS))
    for idx, split in enumerate(SPLITS):
        bottom = 0
        for sex in SEXES:
            count, trails = data[split][sex]
            if count == 0:
                continue
            bar = ax.bar(
                idx,
                count,
                bottom=bottom,
                color=COLORS[split][sex],
                edgecolor="black",
                width=0.6,
            )
            ax.text(
                idx,
                bottom + count / 2,
                f"n=({count},{trails})",
                ha="center",
                va="center",
                color="white",
                fontsize=8,
            )
            bottom += count

    ax.set_xticks(list(x))
    ax.set_xticklabels([s.title() for s in SPLITS])
    ax.set_ylabel("Unique individuals")
    ax.set_title(species)

    out_path = OUT_DIR / f"summary_{species.replace(' ', '_')}.png"
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved {out_path}")


def main() -> None:
    df = load_summary(DATA_PATH)
    for species in sorted(df["Species"].unique()):
        data = summary_for_species(df, species)
        plot_species(species, data)


if __name__ == "__main__":
    main()
