from __future__ import annotations

from pathlib import Path
from ast import literal_eval

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform
from scipy.stats import chi2
from IPython.display import Image, Markdown, display
from sklearn.feature_selection import SelectKBest, f_classif

from FIT_python.caption_utils import save_caption
from FIT_python.data_split_and_summary.data_import_utils import get_feature_cols
from FIT_python.Visualisations.plot_style import SEX_COLORS


def select_top_features(df: pd.DataFrame, target: str, k: int = 4) -> list[str]:
    """Return the ``k`` highest scoring feature names using ANOVA F-test."""

    feature_cols = get_feature_cols(df)
    if not feature_cols:
        return []

    X = df[feature_cols]
    y = df[target]
    if not pd.api.types.is_numeric_dtype(y):
        y = pd.factorize(y)[0]

    skb = SelectKBest(score_func=f_classif, k="all").fit(X, y)
    scores = skb.scores_
    ranked = sorted(zip(feature_cols, scores), key=lambda x: x[1], reverse=True)
    return [feat for feat, _ in ranked[:k]]


def plot_feature_correlation_matrix(
    df: pd.DataFrame, fig_dir: Path, filename: str = "feature_corr_matrix_2x2.png"
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
        "angle": [c for c in num_df.columns if c.lower().startswith("ang")],
        "triangles": [
            c
            for c in num_df.columns
            if c.lower().startswith("t") and not c.lower().startswith("trail")
        ],
    }

    sel = sorted({c for cols in groups.values() for c in cols})
    corr_full = num_df[sel].corr()

    fig, axes = plt.subplots(2, 2, figsize=plt.rcParams["figure.figsize"])
    cmap = "RdBu_r"

    # a) full correlation, no internal colorbar
    ax = axes[0, 0]
    sns.heatmap(
        corr_full,
        cmap=cmap,
        center=0,
        ax=ax,
        xticklabels=False,
        yticklabels=False,
        cbar=False,
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
            corr,
            cmap=cmap,
            center=0,
            ax=ax,
            xticklabels=False,
            yticklabels=False,
            cbar=False,
        )
        n = len(cols)
        mid = (n - 1) / 2
        ax.set_xticks([mid])
        ax.set_xticklabels([f"{grp} ({n})"], rotation=0, fontsize="small")
        ax.set_yticks([mid])
        ax.set_yticklabels([f"{grp} ({n})"], rotation=0, fontsize="small")
        ax.set_title(title, loc="left")

    single(axes[0, 1], "distance", groups["distance"], "b)")
    single(axes[1, 0], "angle", groups["angle"], "c)")
    single(axes[1, 1], "triangles", groups["triangles"], "d)")

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
        out, "2×2 panel of correlation heatmaps a)–d) with a single external colorbar."
    )
    return out


def plot_feature_correlations(
    df: pd.DataFrame, fig_dir: Path, filename: str = "feature_corr_matrix_2x2.png"
) -> Path:
    """Backward compatible wrapper for :func:`plot_feature_correlation_matrix`."""

    return plot_feature_correlation_matrix(df, fig_dir, filename)


def plot_feature_distributions(
    raw_df: pd.DataFrame,
    cleaned_df: pd.DataFrame,
    fig_dir: Path,
    bins: int = 30,
) -> list[Path]:
    """Plot histograms of numeric features before and after cleaning.

    Parameters
    ----------
    raw_df : pandas.DataFrame
        Data prior to cleaning.
    cleaned_df : pandas.DataFrame
        Data after cleaning.
    fig_dir : pathlib.Path
        Directory to store the generated plots.
    bins : int, default=30
        Number of histogram bins.

    Returns
    -------
    list[pathlib.Path]
        Paths of the saved plot images.
    """

    fig_dir.mkdir(parents=True, exist_ok=True)

    feature_cols = get_feature_cols(cleaned_df)
    saved: list[Path] = []
    for col in feature_cols:
        fig, ax = plt.subplots(figsize=plt.rcParams["figure.figsize"])
        sns.histplot(
            raw_df[col].dropna(), bins=bins, color="grey", alpha=0.5, label="raw", ax=ax
        )
        sns.histplot(
            cleaned_df[col].dropna(),
            bins=bins,
            color="blue",
            alpha=0.5,
            label="cleaned",
            ax=ax,
        )
        ax.set_title(col)
        ax.set_xlabel(col)
        ax.set_ylabel("count")
        ax.legend()

        out = fig_dir / f"{col}.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        save_caption(out, f"Histogram of {col} before and after cleaning.")
        saved.append(out)

    return saved


# --- Helper-Funktion für Individual-Boxplots ---
def plot_individual_boxplots(df, top4_feats, fig_dir, filename):
    """
    Zeichnet 2×2 Boxplots der vier Features in top4_feats,
    geordnet nach sex_mapped (erst alle 'Female', dann 'Male'),
    ohne x-Ticks und ohne Legende, speichert das Bild und gibt den Pfad zurück.
    """
    # Bestimme Reihenfolge der individual_id nach Sex
    female_ids = df.loc[df["sex_mapped"] == "Female", "individual_id"].unique().tolist()
    male_ids = df.loc[df["sex_mapped"] == "Male", "individual_id"].unique().tolist()
    ind_order = female_ids + male_ids

    display(Markdown(f"**{filename.replace('.png','')}**"))
    fig, axes = plt.subplots(2, 2, figsize=plt.rcParams["figure.figsize"])
    for ax, feat in zip(axes.flat, top4_feats):
        sns.boxplot(
            data=df,
            x="individual_id",
            y=feat,
            ax=ax,
            hue="sex_mapped",
            palette=SEX_COLORS,
            dodge=False,
            order=ind_order,
            hue_order=["Female", "Male"],
            legend=False,
        )
        ax.set_xlabel("")
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

    fig, axes = plt.subplots(2, 2, figsize=plt.rcParams["figure.figsize"])
    for ax, feat in zip(axes.flat, features):
        sns.boxplot(
            x="sex_mapped",
            y=feat,
            data=plot_df,
            ax=ax,
            hue="sex_mapped",
            palette={k: SEX_COLORS[k] for k in order},
            order=order,
            hue_order=order,
            legend=False,
            dodge=False,
        )
        ax.set_xlabel("Sex")
        ax.set_ylabel(feat)
    plt.tight_layout()
    out = fig_dir / filename
    fig.savefig(out, dpi=150)
    plt.close(fig)

    save_caption(out, f"Boxplots of features {features} by sex (Female, Male).")
    return out


def plot_sex_feature_boxplots(
    df_dict: dict[str, pd.DataFrame],
    fig_dir: Path,
) -> list[Path]:
    """Plot BCR and count difference boxplots for pipelines with/without sex.

    Parameters
    ----------
    df_dict : dict[str, pandas.DataFrame]
        Mapping ``"with_sex"`` and ``"without_sex"`` to summary tables. Each
        table must contain the columns ``species``, ``bcr``, ``pred_count`` and
        ``true_count``.
    fig_dir : pathlib.Path
        Directory to store the generated plots.

    Returns
    -------
    list[pathlib.Path]
        Paths of the generated images in the order
        ``[bcr_species, bcr_all, count_species, count_all]``.
    """

    required_keys = {"with_sex", "without_sex"}
    if not required_keys.issubset(df_dict):
        raise KeyError("df_dict must contain 'with_sex' and 'without_sex'")

    req_cols = {"species", "bcr", "pred_count", "true_count"}
    dfs: list[pd.DataFrame] = []
    for key, df in df_dict.items():
        if not req_cols.issubset(df.columns):
            raise KeyError(f"DataFrame for '{key}' missing required columns")
        tmp = df[list(req_cols)].copy()
        tmp["setup"] = key
        dfs.append(tmp)

    plot_df = pd.concat(dfs, ignore_index=True)
    plot_df["count_diff"] = plot_df["pred_count"] - plot_df["true_count"]

    fig_dir.mkdir(parents=True, exist_ok=True)

    paths: list[Path] = []

    # --- BCR per species ---
    fig, ax = plt.subplots(figsize=plt.rcParams["figure.figsize"])
    sns.boxplot(data=plot_df, x="species", y="bcr", hue="setup", ax=ax)
    ax.set_xlabel("Species")
    ax.set_ylabel("BCR")
    plt.xticks(rotation=45, ha="right")
    fig.tight_layout()
    out = fig_dir / "bcr_by_species.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    save_caption(out, "BCR per species with and without sex features")
    paths.append(out)

    # --- BCR aggregated ---
    fig, ax = plt.subplots(figsize=plt.rcParams["figure.figsize"])
    sns.boxplot(
        data=plot_df, x="setup", y="bcr", ax=ax, order=["with_sex", "without_sex"]
    )
    ax.set_xlabel("Feature set")
    ax.set_ylabel("BCR")
    fig.tight_layout()
    out = fig_dir / "bcr_aggregated.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    save_caption(out, "Aggregated BCR with and without sex features")
    paths.append(out)

    # --- Count difference per species ---
    fig, ax = plt.subplots(figsize=plt.rcParams["figure.figsize"])
    sns.boxplot(data=plot_df, x="species", y="count_diff", hue="setup", ax=ax)
    ax.axhline(0, ls="--", c="gray")
    ax.set_xlabel("Species")
    ax.set_ylabel("pred_count - true_count")
    plt.xticks(rotation=45, ha="right")
    fig.tight_layout()
    out = fig_dir / "countdiff_by_species.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    save_caption(out, "Predicted minus true counts per species")
    paths.append(out)

    # --- Count difference aggregated ---
    fig, ax = plt.subplots(figsize=plt.rcParams["figure.figsize"])
    sns.boxplot(
        data=plot_df,
        x="setup",
        y="count_diff",
        ax=ax,
        order=["with_sex", "without_sex"],
    )
    ax.axhline(0, ls="--", c="gray")
    ax.set_xlabel("Feature set")
    ax.set_ylabel("pred_count - true_count")
    fig.tight_layout()
    out = fig_dir / "countdiff_aggregated.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    save_caption(out, "Aggregated predicted minus true counts")
    paths.append(out)

    return paths


def plot_umap_scatter(
    df: pd.DataFrame,
    emb: pd.DataFrame,
    fig_dir: Path,
    filename: str = "umap_projection.png",
    order: list[str] = ["Female", "Male"],
) -> Path:
    """
    Plot and save a UMAP scatter colored by ``sex_mapped``.

    Parameters
    ----------
    df : pandas.DataFrame
        Unused placeholder for API compatibility.
    emb : pandas.DataFrame
        Must contain the columns ``UMAP1`` and ``UMAP2`` for the coordinates
        as well as ``sex_mapped`` for coloring.

    Returns
    -------
    pathlib.Path
        Path to the saved image file.
    """
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=plt.rcParams["figure.figsize"])
    sns.scatterplot(
        data=emb,
        x="UMAP1",
        y="UMAP2",
        hue="sex_mapped",
        palette={k: SEX_COLORS[k] for k in order},
        hue_order=order,
        alpha=0.7,
        edgecolor="none",
        ax=ax,
    )
    ax.legend(title="Sex")
    plt.tight_layout()
    out = fig_dir / filename
    fig.savefig(out, dpi=150)
    plt.close(fig)

    save_caption(out, "2D UMAP projection colored by sex (Female, Male).")
    return out


# --- 2) Small‑Multiples‑Visualisierung mit Centroid‑Outliern ---
def plot_umap_centroid_outliers(df, title, cols=6):
    n_ind = df["individual_id"].nunique()
    rows = int(np.ceil(n_ind / cols))
    fig, axes = plt.subplots(
        rows, cols, figsize=(cols * 3, rows * 3), sharex=False, sharey=False
    )
    axes = axes.flatten()

    display(Markdown(f"**{title}**"))
    for ax, (ind, subset) in zip(axes, df.groupby("individual_id")):
        # a) KDE-Heatmap der echten Punkte
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
        # b) Alle Punkte
        ax.scatter(
            subset["UMAP1"],
            subset["UMAP2"],
            s=20,
            edgecolor="w",
            linewidth=0.5,
            c=SEX_COLORS[subset["sex_mapped"].iloc[0]],
        )
        # c) Centroid‑Outlier als rote Kreuze
        out = subset[subset["is_outlier_centroid"]]
        if not out.empty:
            ax.scatter(
                out["UMAP1"],
                out["UMAP2"],
                marker="x",
                c="red",
                s=40,
                label="Outlier (Centroid)",
            )
        ax.set_title(ind)
        ax.set_xlabel("UMAP1")
        ax.set_ylabel("UMAP2")

    # Leere Plots abschalten
    for ax in axes[n_ind:]:
        ax.axis("off")

    # Gemeinsame Legende
    handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="gray",
            markersize=8,
            label="Inlier",
        ),
        plt.Line2D(
            [0], [0], marker="x", color="red", markersize=8, label="Outlier (Centroid)"
        ),
    ]
    fig.legend(handles=handles, loc="upper right")
    plt.tight_layout(rect=[0, 0, 0.95, 1])
    return fig


def _parse_coords(val) -> np.ndarray:
    """Return ``val`` as a 1D float array.

    ``val`` may either be a sequence of numbers or a string representation
    of such a sequence. Invalid inputs yield an empty array.
    """

    if isinstance(val, str):
        try:
            arr = np.asarray(literal_eval(val), dtype=float)
        except Exception:
            return np.empty(0, dtype=float)
    else:
        arr = np.asarray(val, dtype=float)
    return arr


def _circle_from_row(
    row: pd.Series, prefix: str
) -> tuple[tuple[float, float], float] | tuple[None, None]:
    """Return center and radius for the ``prefix`` coordinates in ``row``."""

    x = _parse_coords(row.get(f"coords_{prefix}_x"))
    y = _parse_coords(row.get(f"coords_{prefix}_y"))
    if x.size == 0 or y.size == 0 or x.size != y.size:
        return None, None
    pts = np.column_stack([x, y])
    center = pts.mean(axis=0)
    dist = np.linalg.norm(pts - center, axis=1)
    std = dist.std(ddof=1) if dist.size > 1 else 0.0
    radius = np.sqrt(chi2.ppf(0.5, df=2)) * std
    return (float(center[0]), float(center[1])), float(radius)


def _rhombus_from_row(
    row: pd.Series, prefix: str
) -> tuple[tuple[float, float], float] | tuple[None, None]:
    """Return center and L1-based radius for the ``prefix`` coordinates in ``row``."""

    x = _parse_coords(row.get(f"coords_{prefix}_x"))
    y = _parse_coords(row.get(f"coords_{prefix}_y"))
    if x.size == 0 or y.size == 0 or x.size != y.size:
        return None, None
    pts = np.column_stack([x, y])
    center = pts.mean(axis=0)
    dist = np.abs(pts - center).sum(axis=1)
    std = dist.std(ddof=1) if dist.size > 1 else 0.0
    radius = chi2.ppf(0.5, df=2) * std
    return (float(center[0]), float(center[1])), float(radius)


def plot_pair_examples(
    df_res: pd.DataFrame, out_dir: Path, *, rhombus: bool = False
) -> list[Path]:
    """Plot example 50% confidence areas for TP/FP/FN/TN categories.

    Parameters
    ----------
    df_res : pandas.DataFrame
        Result rows containing coordinate columns.
    out_dir : pathlib.Path
        Directory where the output image is written.
    rhombus : bool, default=False
        If ``True``, draw rhombus confidence regions based on Manhattan
        distances instead of circular ones.
    """

    out_dir.mkdir(parents=True, exist_ok=True)

    mapping = {"True": True, "False": False, True: True, False: False}
    df = df_res.copy()
    df["_gt"] = df["same_individual"].map(mapping)
    df["_pred"] = df["pred"].map(mapping)

    cats = {
        "TP": (True, True),
        "FP": (False, True),
        "FN": (True, False),
        "TN": (False, False),
    }

    examples: dict[str, pd.Series] = {}
    for name, (gt, pr) in cats.items():
        sel = df[(df["_gt"] == gt) & (df["_pred"] == pr)]
        if not sel.empty:
            examples[name] = sel.iloc[0]

    paths: list[Path] = []
    for key in ("TP", "FP", "FN", "TN"):
        fig, ax = plt.subplots(figsize=(4, 4))
        row = examples.get(key)
        if row is None:
            ax.axis("off")
        else:
            if rhombus:
                c_a, r_a = _rhombus_from_row(row, "a")
                c_b, r_b = _rhombus_from_row(row, "b")
            else:
                c_a, r_a = _circle_from_row(row, "a")
                c_b, r_b = _circle_from_row(row, "b")

            if c_a is not None:
                if rhombus:
                    coords = [
                        (c_a[0] + r_a, c_a[1]),
                        (c_a[0], c_a[1] + r_a),
                        (c_a[0] - r_a, c_a[1]),
                        (c_a[0], c_a[1] - r_a),
                    ]
                    ax.add_patch(Polygon(coords, fill=False, color="blue", lw=2))
                else:
                    ax.add_patch(plt.Circle(c_a, r_a, fill=False, color="blue", lw=2))
            if c_b is not None:
                if rhombus:
                    coords = [
                        (c_b[0] + r_b, c_b[1]),
                        (c_b[0], c_b[1] + r_b),
                        (c_b[0] - r_b, c_b[1]),
                        (c_b[0], c_b[1] - r_b),
                    ]
                    ax.add_patch(Polygon(coords, fill=False, color="orange", lw=2))
                else:
                    ax.add_patch(plt.Circle(c_b, r_b, fill=False, color="orange", lw=2))

        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(key)
        fig.tight_layout()

        suffix = "_rhombus" if rhombus else ""
        out_path = out_dir / f"pair_example_{key}{suffix}.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)

        caption = (
            f"50% confidence rhombus for {key} example pair."
            if rhombus
            else f"50% confidence circle for {key} example pair."
        )
        save_caption(out_path, caption)
        paths.append(out_path)

    return paths


def plot_dendrogram(dist_matrix: pd.DataFrame, cutoff: float, out_file: Path) -> Path:
    """Save a dendrogram based on ``dist_matrix`` with a cutoff line."""

    out_file.parent.mkdir(parents=True, exist_ok=True)
    condensed = squareform(dist_matrix.to_numpy(), checks=False)
    link = linkage(condensed, method="ward")

    fig, ax = plt.subplots(figsize=plt.rcParams["figure.figsize"])
    dendrogram(link, labels=dist_matrix.index.astype(str).tolist(), ax=ax)
    ax.axhline(cutoff, color="red", linestyle="--")
    ax.set_ylabel("Ward distance")
    ax.set_xlabel("Sample")
    plt.tight_layout()
    fig.savefig(out_file, dpi=150)
    plt.close(fig)
    save_caption(out_file, "Ward dendrogram with distance cutoff line.")
    return out_file


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

    base = [
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
    fig.legend(
        handles=legend_elems,
        loc="center right",
        bbox_to_anchor=(1.15, 0.5),
        title="Sex",
    )
    fig.tight_layout(rect=[0, 0, 0.85, 1])

    out = fig_dir / filename
    fig.savefig(out, dpi=150)
    plt.close(fig)
    save_caption(out, f"{xcol}/{ycol} scatter by individual")
    return out


def plot_umap_by_individual(
    df: pd.DataFrame,
    marker_map: dict,
    fig_dir: Path,
    filename: str,
    xcol: str = "UMAP1",
    ycol: str = "UMAP2",
) -> Path:
    """Scatter UMAP coordinates coloured by sex with markers per individual."""

    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    fig_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 6))  # etwas kompakter, aber gute Lesbarkeit

    for ind, subset in df.groupby("individual_id"):
        _scatter_points(
            ax,
            subset[xcol],
            subset[ycol],
            SEX_COLORS[subset["sex_mapped"].iloc[0]],
            marker_map.get(ind, "o"),
            label=str(ind),
        )

    ax.set_xlabel(xcol)
    ax.set_ylabel(ycol)

    # Simplified legend showing individuals grouped by sex
    female_ids = sorted(
        df.loc[df["sex_mapped"] == "Female", "individual_id"].unique().tolist()
    )
    male_ids = sorted(
        df.loc[df["sex_mapped"] == "Male", "individual_id"].unique().tolist()
    )

    female_handles = [
        Line2D(
            [0],
            [0],
            marker=marker_map.get(ind, "o"),
            color=SEX_COLORS["Female"],
            linestyle="",
            markersize=6,
            label=str(ind),
        )
        for ind in female_ids
    ]
    male_handles = [
        Line2D(
            [0],
            [0],
            marker=marker_map.get(ind, "o"),
            color=SEX_COLORS["Male"],
            linestyle="",
            markersize=6,
            label=str(ind),
        )
        for ind in male_ids
    ]

    legend_f = ax.legend(
        handles=female_handles,
        title="Female",
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0,
        fontsize="small",
        title_fontsize="medium",
    )
    legend_m = ax.legend(
        handles=male_handles,
        title="Male",
        loc="upper left",
        bbox_to_anchor=(1.20, 1.0),
        borderaxespad=0,
        fontsize="small",
        title_fontsize="medium",
    )
    ax.add_artist(legend_f)
    ax.add_artist(legend_m)

    fig.tight_layout(rect=[0, 0, 0.75, 1])  # mehr Platz für Plot, weniger für Legenden

    out = fig_dir / filename
    fig.savefig(out, dpi=150)
    plt.close(fig)
    save_caption(out, f"{xcol}/{ycol} UMAP by individual and sex")
    return out


def plot_umap_centroid_outliers(
    df: pd.DataFrame, title: str, cols: int = 6
) -> plt.Figure:
    """Small-multiple plots of UMAP points with centroid-based outliers."""
    n_ind = df["individual_id"].nunique()
    rows = int(np.ceil(n_ind / cols))
    fig, axes = plt.subplots(
        rows, cols, figsize=(cols * 3, rows * 3), sharex=False, sharey=False
    )
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
            ax.scatter(
                out["UMAP1"],
                out["UMAP2"],
                marker="x",
                c="red",
                s=40,
                label="Outlier (Centroid)",
            )
        ax.set_title(ind)
        ax.set_xlabel("UMAP1")
        ax.set_ylabel("UMAP2")

    for ax in axes[n_ind:]:
        ax.axis("off")

    handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="gray",
            markersize=8,
            label="Inlier",
        ),
        plt.Line2D(
            [0], [0], marker="x", color="red", markersize=8, label="Outlier (Centroid)"
        ),
    ]
    fig.legend(handles=handles, loc="upper right")
    plt.tight_layout(rect=[0, 0, 0.95, 1])
    return fig


def plot_umap_centroid_outliers(
    df: pd.DataFrame, title: str, cols: int = 6
) -> plt.Figure:
    """
    Small‑multiple UMAP plots with KDE background and optional centroid‑based outlier markers.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns 'UMAP1', 'UMAP2', 'individual_id', 'sex_mapped',
        and optionally 'is_outlier_centroid'.
    title : str
        Title to display above the grid.
    cols : int, default=6
        Number of columns in the facet grid.

    Returns
    -------
    fig : plt.Figure
        The matplotlib Figure object containing the small multiples.
    """
    n_ind = df["individual_id"].nunique()
    rows = int(np.ceil(n_ind / cols))
    fig, axes = plt.subplots(
        rows, cols, figsize=(cols * 3, rows * 3), sharex=False, sharey=False
    )
    axes = axes.flatten()

    display(Markdown(f"**{title}**"))

    # Detect if outlier column is present
    has_outliers = "is_outlier_centroid" in df.columns

    for ax, (ind, subset) in zip(axes, df.groupby("individual_id")):
        # 1) KDE‑Heatmap of the true points
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

        # 2) Scatter all points
        ax.scatter(
            subset["UMAP1"],
            subset["UMAP2"],
            s=20,
            edgecolor="w",
            linewidth=0.5,
            c=SEX_COLORS[subset["sex_mapped"].iloc[0]],
        )

        # 3) Mark centroid‑based outliers if present
        if has_outliers:
            out = subset[subset["is_outlier_centroid"]]
            if not out.empty:
                ax.scatter(
                    out["UMAP1"],
                    out["UMAP2"],
                    marker="x",
                    c="red",
                    s=40,
                    label="Outlier (Centroid)",
                )

        ax.set_title(ind)
        ax.set_xlabel("UMAP1")
        ax.set_ylabel("UMAP2")

    # Turn off any unused subplots
    for ax in axes[n_ind:]:
        ax.axis("off")

    # Add a global legend if outliers are shown
    if has_outliers:
        handles = [
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor="gray",
                markersize=8,
                label="Inlier",
            ),
            plt.Line2D(
                [0],
                [0],
                marker="x",
                color="red",
                markersize=8,
                label="Outlier (Centroid)",
            ),
        ]
        fig.legend(handles=handles, loc="upper right")

    plt.tight_layout(rect=[0, 0, 0.95, 1])
    return fig
