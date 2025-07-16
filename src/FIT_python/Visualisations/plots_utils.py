from __future__ import annotations

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from FIT_python.Visualisations.plot_style import SEX_COLORS
from FIT_python.caption_utils import save_caption

def plot_feature_correlation_matrix(
    df: pd.DataFrame,
    fig_dir: Path,
    filename: str = "feature_corr_matrix_2x2.png"
) -> Path:
    """
    Create and save a 2×2 panel of correlation heatmaps:
     - a) full correlation (distance/angle/triangles)
     - b) distance-only features
     - c) angle-only features
     - d) triangles-only features
    Returns the path to the saved image.
    """
    fig_dir.mkdir(parents=True, exist_ok=True)

    num_df = df.select_dtypes(include="number")
    groups = {
        "distance": [c for c in num_df.columns if c.lower().startswith(("d","dist"))],
        "angle":    [c for c in num_df.columns if c.lower().startswith("ang")],
        "triangles":[c for c in num_df.columns if c.lower().startswith("t") and not c.lower().startswith("trail")]
    }

    sel = sorted({c for cols in groups.values() for c in cols})
    corr_full = num_df[sel].corr()

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    cmap = "RdBu_r"

    ax = axes[0, 0]
    sns.heatmap(corr_full, cmap=cmap, center=0, ax=ax,
                xticklabels=False, yticklabels=False,
                cbar_kws={"shrink": 0.8})
    ordered = corr_full.columns.tolist()
    positions, labels = [], []
    for grp, cols in groups.items():
        idxs = [ordered.index(c) for c in cols if c in ordered]
        if not idxs:
            continue
        mid = (min(idxs) + max(idxs)) / 2
        positions.append(mid)
        labels.append(f"{grp} ({len(idxs)})")
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=0, fontsize="small")
    ax.set_yticks(positions)
    ax.set_yticklabels(labels, rotation=0, fontsize="small")
    ax.set_title("a)", loc="left")

    def single(ax, grp, cols, title):
        corr = num_df[cols].corr()
        sns.heatmap(corr, cmap=cmap, center=0, ax=ax,
                    xticklabels=False, yticklabels=False,
                    cbar=False)
        n = len(cols)
        mid = (n - 1) / 2
        ax.set_xticks([mid])
        ax.set_xticklabels([f"{grp} ({n})"], rotation=0, fontsize="small")
        ax.set_yticks([mid])
        ax.set_yticklabels([f"{grp} ({n})"], rotation=0, fontsize="small")
        ax.set_title(title, loc="left")

    single(axes[0,1], "distance", groups["distance"], "b)")
    single(axes[1,0], "angle",    groups["angle"],    "c)")
    single(axes[1,1], "triangles",groups["triangles"],"d)")

    plt.tight_layout()
    out = fig_dir / filename
    fig.savefig(out, dpi=150)
    plt.close(fig)

    save_caption(
        out,
        "2×2 panel of correlation heatmaps: a) full, b) distance, c) angle, d) triangles."
    )
    return out

def plot_sex_boxplots(
    df: pd.DataFrame,
    features: list[str],
    fig_dir: Path,
    filename: str = "boxplots_top4_features.png",
    order: list[str] = ["Female", "Male"],
) -> Path:
    """
    Plot and save boxplots for given features split by sex (Female/Male).
    Filters out Unknown.
    """
    fig_dir.mkdir(parents=True, exist_ok=True)
    plot_df = df[df["sex_mapped"].isin(order)]

    fig, axes = plt.subplots(2, 2, figsize=(8, 6))
    for ax, feat in zip(axes.flat, features):
        sns.boxplot(
            x="sex_mapped", y=feat, data=plot_df, ax=ax,
            palette={k: SEX_COLORS[k] for k in order},
            order=order
        )
        ax.set_xlabel("Sex")
        ax.set_ylabel(feat)
    plt.tight_layout()
    out = fig_dir / filename
    fig.savefig(out, dpi=150)
    plt.close(fig)

    save_caption(
        out,
        f"Boxplots of features {features} by sex (Female, Male)."
    )
    return out

def plot_umap_scatter(
    df: pd.DataFrame,
    emb: pd.DataFrame,
    fig_dir: Path,
    filename: str = "umap_projection.png",
    order: list[str] = ["Female", "Male"],
) -> Path:
    """
    Plot and save a UMAP scatter colored by sex mapped.
    emb: DataFrame with columns ['UMAP1','UMAP2'].
    """
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.scatterplot(
        data=emb, x="UMAP1", y="UMAP2",
        hue="sex_mapped",
        palette={k:SEX_COLORS[k] for k in order},
        hue_order=order,
        alpha=0.7,
        ax=ax
    )
    ax.legend(title="Sex")
    plt.tight_layout()
    out = fig_dir / filename
    fig.savefig(out, dpi=150)
    plt.close(fig)

    save_caption(
        out,
        "2D UMAP projection colored by sex (Female, Male)."
    )
    return out
