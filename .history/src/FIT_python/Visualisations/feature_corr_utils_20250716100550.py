from __future__ import annotations

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from FIT_python.plot_style import apply_style
from FIT_python.caption_utils import save_caption


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

from FIT_python.plot_style import apply_style
from FIT_python.caption_utils import save_caption

def plot_feature_correlation_matrix(df: pd.DataFrame, fig_dir: Path) -> Path:
    """
    Create a 2×2 panel figure:
     - a) full correlation (distance/angle/triangles)
     - b) correlation only among 'distance' features
     - c) correlation only among 'angle' features
     - d) correlation only among 'triangles' features
    For the single‐group panels, only a single tick is drawn at the block‐center
    with label 'group_name (n_features)'.
    """
    apply_style()
    fig_dir.mkdir(exist_ok=True, parents=True)

    num_df = df.select_dtypes(include="number")
    groups = {
        "distance": [c for c in num_df if c.lower().startswith(("d","dist"))],
        "angle":    [c for c in num_df if c.lower().startswith("ang")],
        "triangles":[c for c in num_df if c.lower().startswith("t") and not c.lower().startswith("trail")]
    }

    # panel‐a: full heatmap
    sel = sorted({c for cols in groups.values() for c in cols})
    corr_full = num_df[sel].corr()

    # prepare figure
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    cmap="RdBu_r"
    # --- Panel a) full groups heatmap ---
    ax = axes[0,0]
    sns.heatmap(corr_full, cmap=cmap, center=0, ax=ax, cbar_kws={"shrink":0.8}, 
                xticklabels=False, yticklabels=False)
    # compute group tick positions as before
    ordered = corr_full.columns.tolist()
    positions, labels = [], []
    for grp, cols in groups.items():
        idxs = [ordered.index(c) for c in cols if c in ordered]
        if not idxs: continue
        mid = (min(idxs)+max(idxs))/2
        positions.append(mid); labels.append(f"{grp} ({len(idxs)})")
    ax.set_xticks(positions); ax.set_xticklabels(labels, rotation=0, fontsize="small")
    ax.set_yticks(positions); ax.set_yticklabels(labels, rotation=0, fontsize="small")
    ax.set_title("a)", loc="left")

    # helper to draw single‐group panels
    def single_group_panel(ax, grp_label, cols, title):
        corr = num_df[cols].corr()
        sns.heatmap(corr, cmap=cmap, center=0, ax=ax,
                    xticklabels=False, yticklabels=False, cbar=False)
        n = len(cols)
        mid = (n-1)/2
        ax.set_xticks([mid])
        ax.set_xticklabels([f"{grp_label} ({n})"], rotation=0, fontsize="small")
        ax.set_yticks([mid])
        ax.set_yticklabels([f"{grp_label} ({n})"], rotation=0, fontsize="small")
        ax.set_title(title, loc="left")

    # Panel b) distance
    single_group_panel(axes[0,1], "distance", groups["distance"], "b)")
    # Panel c) angle
    single_group_panel(axes[1,0], "angle",    groups["angle"],    "c)")
    # Panel d) triangles
    single_group_panel(axes[1,1], "triangles",groups["triangles"],"d)")

    plt.tight_layout()
    out = fig_dir / "feature_corr_matrix_2x2.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)

    save_caption(out,
                 "2×2 panel of correlation heatmaps: a) full, "
                 "b) distance, c) angle, d) triangles.")
    return out
