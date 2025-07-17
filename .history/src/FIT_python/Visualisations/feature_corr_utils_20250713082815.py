from __future__ import annotations

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from FIT_python.plot_style import apply_style
from FIT_python.caption_utils import save_caption


def plot_feature_correlations(df: pd.DataFrame, fig_dir: Path) -> Path:
    """Plot a correlation heatmap for feature groups.

    Groups are determined by feature prefixes:
    - ``distance``: columns starting with ``d`` or ``dist``
    - ``angle``: columns starting with ``ang``
    - ``triangles``: columns starting with ``t`` (excluding ``trail``)

    If at least two of these groups are present, the columns of each
    group are averaged per row and a correlation matrix between the
    groups is plotted. Otherwise the correlation matrix of all numeric
    features is shown without axis labels.
    """
    apply_style()
    fig_dir.mkdir(parents=True, exist_ok=True)

    num_df = df.select_dtypes(include="number")
    groups = {
        "distance": [c for c in num_df.columns if c.lower().startswith("d") or c.lower().startswith("dist")],
        "angle": [c for c in num_df.columns if c.lower().startswith("ang")],
        "triangles": [c for c in num_df.columns if c.lower().startswith("t") and not c.lower().startswith("trail")],
    }

    data = {name: num_df[cols].mean(axis=1) for name, cols in groups.items() if cols}
    use_groups = len(data) >= 2

    if use_groups:
        corr = pd.DataFrame(data).corr()
    else:
        corr = num_df.corr()

    fig, ax = plt.subplots(figsize=(4, 3))
    sns.heatmap(corr, cmap="viridis", center=0, ax=ax, cbar_kws={"shrink": 0.8})
    if use_groups:
        ax.set_xticklabels(corr.columns, rotation=45, ha="right")
        ax.set_yticklabels(corr.index, rotation=0)
    else:
        ax.set_xticks([])
        ax.set_yticks([])
    ax.set_title("Feature Correlation")
    fig.tight_layout()

    out_file = fig_dir / "feature_corr_heatmap.png"
    fig.savefig(out_file)
    caption = (
        "Correlation between feature groups" if use_groups else "Feature correlation heatmap"
    )
    save_caption(out_file, caption)
    plt.close(fig)
    return out_file
