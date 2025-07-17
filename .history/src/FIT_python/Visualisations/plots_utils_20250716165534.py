from __future__ import annotations

from pathlib import Path
import numpy as np
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
    Create and save a 2×2 panel of correlation heatmaps with a single, external colorbar:
     - a) full correlation (distance/angle/triangles)
     - b) distance-only features
     - c) angle-only features
     - d) triangles-only features
    Returns the path to the saved image.
    """
    import matplotlib as mpl

    fig_dir.mkdir(parents=True, exist_ok=True)

    num_df = df.select_dtypes(include="number")
    groups = {
        "distance": [c for c in num_df.columns if c.lower().startswith(("d", "dist"))],
        "angle":    [c for c in num_df.columns if c.lower().startswith("ang")],
        "triangles":[c for c in num_df.columns if c.lower().startswith("t") and not c.lower().startswith("trail")]
    }

    sel = sorted({c for cols in groups.values() for c in cols})
    corr_full = num_df[sel].corr()

    fig, axes = plt.subplots(2, 2, figsize=plt.rcParams['figure.figsize'])
    cmap = "RdBu_r"

    # a) full correlation, no internal colorbar
    ax = axes[0, 0]
    sns.heatmap(
        corr_full, cmap=cmap, center=0, ax=ax,
        xticklabels=False, yticklabels=False, cbar=False
    )
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

    # helper for single-group plots
    def single(ax, grp, cols, title):
        corr = num_df[cols].corr()
        sns.heatmap(
            corr, cmap=cmap, center=0, ax=ax,
            xticklabels=False, yticklabels=False, cbar=False
        )
        n = len(cols)
        mid = (n - 1) / 2
        ax.set_xticks([mid])
        ax.set_xticklabels([f"{grp} ({n})"], rotation=0, fontsize="small")
        ax.set_yticks([mid])
        ax.set_yticklabels([f"{grp} ({n})"], rotation=0, fontsize="small")
        ax.set_title(title, loc="left")

    single(axes[0,1], "distance",  groups["distance"],  "b)")
    single(axes[1,0], "angle",     groups["angle"],     "c)")
    single(axes[1,1], "triangles", groups["triangles"], "d)")

    plt.tight_layout()

    # add a single external colorbar
    cax = fig.add_axes([1.03, 0.15, 0.02, 0.7])
    norm = mpl.colors.Normalize(vmin=-1, vmax=1)
    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, cax=cax, label="Pearson $r$")

    out = fig_dir / filename
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)

    save_caption(
        out,
        "2×2 panel of correlation heatmaps a)–d) with a single external colorbar."
    )
    return out


# --- Helper-Funktion für Individual-Boxplots ---
def plot_individual_boxplots(df, top4_feats, fig_dir, filename):
    """
    Zeichnet 2×2 Boxplots der vier Features in top4_feats,
    geordnet nach sex_mapped (erst alle 'Female', dann 'Male'),
    ohne x-Ticks und ohne Legende, speichert das Bild und gibt den Pfad zurück.
    """
    # Bestimme Reihenfolge der individual_id nach Sex
    female_ids = df.loc[df['sex_mapped']=='Female', 'individual_id'].unique().tolist()
    male_ids   = df.loc[df['sex_mapped']=='Male',   'individual_id'].unique().tolist()
    ind_order  = female_ids + male_ids

    display(Markdown(f"**{filename.replace('.png','')}**"))
    fig, axes = plt.subplots(2, 2, figsize=plt.rcParams['figure.figsize'])
    for ax, feat in zip(axes.flat, top4_feats):
        sns.boxplot(
            data=df,
            x='individual_id', y=feat, ax=ax,
            hue='sex_mapped',
            palette=SEX_COLORS,
            dodge=False,
            order=ind_order,
            hue_order=["Female", "Male"],
            legend=False
        )
        ax.set_xlabel('')
        ax.set_xticks([])
        ax.set_ylabel(feat)
    plt.tight_layout()

    save_path = fig_dir / filename
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    display(Image(filename=str(save_path)))
    return save_path


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

    fig, axes = plt.subplots(2, 2, figsize=plt.rcParams['figure.figsize'])
    for ax, feat in zip(axes.flat, features):
        sns.boxplot(
            x="sex_mapped", y=feat, data=plot_df, ax=ax,
            hue="sex_mapped", palette={k: SEX_COLORS[k] for k in order},
            order=order, hue_order=order, legend=False, dodge=False
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
    fig, ax = plt.subplots(figsize=plt.rcParams['figure.figsize'])
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


FILLED_MARKERS = {"o", "s", "^", "v", "P", "X", "D", "*", "h", "8"}


def _scatter_points(ax, x, y, color, marker, label="") -> None:
    """Helper to plot scatter points with consistent styling."""
    params = dict(c=color, marker=marker, label=label, alpha=0.7)
    if marker in FILLED_MARKERS:
        params.update(edgecolor="w", linewidth=0.5)
    ax.scatter(x, y, **params)


def make_marker_map(ids) -> dict:
    """Assign a distinct marker to each ID."""
    from itertools import cycle

    base = ["o", "s", "^", "v", "P", "X", "D", "*", "h", "+", "x", "1", "2", "3", "4", "8"]
    return {i: m for i, m in zip(ids, cycle(base))}


def plot_individual_boxplots(
    df: pd.DataFrame,
    top4_feats: list[str],
    fig_dir: Path,
    filename: str,
) -> Path:
    """Plot 2×2 boxplots grouped by individual and sex."""
    fig_dir.mkdir(parents=True, exist_ok=True)

    female_ids = df.loc[df["sex_mapped"] == "Female", "individual_id"].unique().tolist()
    male_ids = df.loc[df["sex_mapped"] == "Male", "individual_id"].unique().tolist()
    ind_order = female_ids + male_ids

    fig, axes = plt.subplots(2, 2, figsize=plt.rcParams["figure.figsize"])
    for ax, feat in zip(axes.flat, top4_feats):
        sns.boxplot(
            data=df,
            x="individual_id",
            y=feat,
            hue="sex_mapped",
            palette=SEX_COLORS,
            dodge=False,
            order=ind_order,
            hue_order=["Female", "Male"],
            legend=False,
            ax=ax,
        )
        ax.set_xlabel("")
        ax.set_xticks([])
        ax.set_ylabel(feat)
    plt.tight_layout()

    out = fig_dir / filename
    plt.savefig(out, dpi=150)
    plt.close()
    save_caption(out, f"Boxplots of {top4_feats} grouped by individual")
    return out


def plot_embedding_by_individual(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    marker_map: dict,
    fig_dir: Path,
    filename: str,
    xcol: str,
    ycol: str,
    titles: tuple[str, str],
) -> Path:
    """Scatter embeddings for train/test splits grouped by individual."""
    from matplotlib.patches import Patch

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, df, title in zip(axes, (train_df, test_df), titles):
        for ind, subset in df.groupby("individual_id"):
            _scatter_points(
                ax,
                subset[xcol],
                subset[ycol],
                SEX_COLORS[subset["sex_mapped"].iloc[0]],
                marker_map.get(ind, "o"),
                ind if ax is axes[1] else "",
            )
        ax.set_title(title, loc="left")
        ax.set_xlabel(xcol)
        ax.set_ylabel(ycol)

    legend_elems = [
        Patch(color=SEX_COLORS["Female"], label="Female"),
        Patch(color=SEX_COLORS["Male"], label="Male"),
    ]
    fig.legend(handles=legend_elems, loc="center right", bbox_to_anchor=(1.15, 0.5), title="Sex")
    fig.tight_layout(rect=[0, 0, 0.85, 1])

    out = fig_dir / filename
    fig.savefig(out, dpi=150)
    plt.close(fig)
    save_caption(out, f"{xcol}/{ycol} scatter by individual")
    return out


def mark_outliers_centroid(df: pd.DataFrame, bandwidth: float = 0.5, percentile: float = 1) -> pd.DataFrame:
    """Mark centroid-based outliers per individual in a UMAP embedding."""
    from scipy.stats import multivariate_normal
    df = df.copy()
    df["centroid_density"] = np.nan
    df["is_outlier_centroid"] = False

    for ind, idx in df.groupby("individual_id").groups.items():
        pts = df.loc[idx, ["UMAP1", "UMAP2"]].values
        cx, cy = pts.mean(axis=0)
        cov = [[bandwidth ** 2, 0], [0, bandwidth ** 2]]
        kernel = multivariate_normal(mean=[cx, cy], cov=cov)
        dens = kernel.pdf(pts)
        thresh = np.percentile(dens, percentile)
        df.loc[idx, "centroid_density"] = dens
        df.loc[idx, "is_outlier_centroid"] = dens < thresh
    return df


def plot_umap_centroid_outliers(df: pd.DataFrame, title: str, cols: int = 6) -> plt.Figure:
    """Small-multiple plots of UMAP points with centroid-based outliers."""
    n_ind = df["individual_id"].nunique()
    rows = int(np.ceil(n_ind / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3), sharex=False, sharey=False)
    axes = axes.flatten()

    for ax, (ind, subset) in zip(axes, df.groupby("individual_id")):
        sns.kdeplot(
            data=subset,
            x="UMAP1",
            y="UMAP2",
            fill=True,
            thresh=0.05,
            levels=5,
            alpha=0.4,
            ax=ax,
            color=SEX_COLORS[subset["sex_mapped"].iloc[0]],
        )
        _scatter_points(
            ax,
            subset["UMAP1"],
            subset["UMAP2"],
            SEX_COLORS[subset["sex_mapped"].iloc[0]],
            "o",
        )
        out = subset[subset["is_outlier_centroid"]]
        if not out.empty:
            ax.scatter(out["UMAP1"], out["UMAP2"], marker="x", c="red", s=40, label="Outlier (Centroid)")
        ax.set_title(ind)
        ax.set_xlabel("UMAP1")
        ax.set_ylabel("UMAP2")

    for ax in axes[n_ind:]:
        ax.axis("off")

    handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="gray", markersize=8, label="Inlier"),
        plt.Line2D([0], [0], marker="x", color="red", markersize=8, label="Outlier (Centroid)"),
    ]
    fig.legend(handles=handles, loc="upper right")
    plt.tight_layout(rect=[0, 0, 0.95, 1])
    return fig
