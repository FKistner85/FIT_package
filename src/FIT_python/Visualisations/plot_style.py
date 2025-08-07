from __future__ import annotations

import os

import pandas as pd
import matplotlib as mpl
import seaborn as sns

from FIT_python.config import CONFIG


def map_sex(value):
    """Map raw sex values to standardized labels.

    Also handles pandas Series by mapping elementwise.
    """

    mapping = CONFIG["visualisation"]["sex"]["value_map"]
    if isinstance(value, pd.Series):
        return value.map(lambda v: mapping.get(v, "Unknown"))
    return mapping.get(value, "Unknown")


def apply_style() -> None:
    """Apply consistent plot styling across notebooks and modules."""

    colors = CONFIG["visualisation"]["sex"]["colors"]

    # 1) Apply Seaborn theme with sex-specific palette
    sns.set_theme(style="whitegrid", palette=list(colors.values()))

    # 2) Update Matplotlib rc parameters for consistent styling
    mpl.rcParams.update({
        "figure.dpi": 100,
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.figsize": tuple(map(float, os.getenv("FIGSIZE", "8,6").split(','))),
    })
