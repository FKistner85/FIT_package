from __future__ import annotations

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from FIT_python.Visualisations.plot_style import SEX_COLORS, apply_style
from FIT_python.caption_utils import save_caption

def plot_feature_correlation_matrix(df: pd.DataFrame, fig_dir: Path, filename: str = "feature_correlation_matrix.png") -> Path:
    """Plot a 2x2 grid of correlation heatmaps with a single colorbar.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe containing numeric features.
    fig_dir : pathlib.Path
        Directory where the figure will be saved.
    filename : str, optional
        Name of the output image file, by default ``"feature_correlation_matrix.png"``.

    Returns
    -------
    pathlib.Path
        Path to the saved figure.
    """
    apply_style()
    fig_dir.mkdir(parents=True, exist_ok=True)

    corr = df.corr()
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    vmin, vmax = -1, 1

    fig, axes = plt.subplots(2, 2, figsize=(8, 6))
    for ax in axes.ravel():
        sns.heatmap(corr, cmap=cmap, center=0, vmin=vmin, vmax=vmax, ax=ax, cbar=False)
        ax.set_xticks([])
        ax.set_yticks([])

    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes.ravel().tolist(), fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel("Correlation")

    fig.tight_layout()
    out_file = fig_dir / filename
    fig.savefig(out_file)
    save_caption(out_file, "Correlation heatmaps with shared colorbar")
    plt.close(fig)
    return out_file



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
