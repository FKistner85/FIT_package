from __future__ import annotations

import matplotlib as mpl
import seaborn as sns


def apply_style() -> None:
    """Apply consistent plot styling across notebooks and modules."""
    sns.set_theme(style="whitegrid", palette="colorblind")

    mpl.rcParams.update({
        "figure.dpi": 100,
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
    })

