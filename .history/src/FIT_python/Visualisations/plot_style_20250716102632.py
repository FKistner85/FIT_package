from __future__ import annotations

import matplotlib as mpl
import seaborn as sns

# Base colours used for all sex-specific plots
SEX_COLORS = {"F": "#800000", "M": "#000080", "Unknown": "#FFA500"}  # orange


def _lighten(color: str, amount: float) -> str:
    """Return a lighter shade of ``color``.

    ``amount`` specifies the blend ratio with white where ``0`` returns the
    original colour and ``1`` returns white.
    """
    r, g, b = mpl.colors.to_rgb(color)
    return mpl.colors.to_hex(
        [
            r + (1 - r) * amount,
            g + (1 - g) * amount,
            b + (1 - b) * amount,
        ]
    )


TRAIN_COLORS = SEX_COLORS
TEST_COLORS = {k: _lighten(v, 0.5) for k, v in SEX_COLORS.items()}


# Updated apply_style function including sex mapping and colors

import matplotlib as mpl
import seaborn as sns

# Enhanced apply_style with sex mapping and colors

import matplotlib as mpl
import seaborn as sns

def apply_style() -> None:
    """Apply consistent plot styling across notebooks and modules,
    including sex-specific mappings and color palette.
    
    Raw sex values 'f', 'F', 0 -> 'Female'; 
    'm', 'M', 1 -> 'Male'; others -> 'Unknown'.
    """

    # 1) Define raw-to-label mapping
    global SEX_VALUE_MAP
    SEX_VALUE_MAP = {
        'f': 'Female', 'F': 'Female', 0: 'Female',
        'm': 'Male',   'M': 'Male',   1: 'Male'
    }

    # 2) Define sex-specific color palette
    global SEX_COLORS
    SEX_COLORS = {
        "Female": "#800000",  # dark red
        "Male":   "#000080",  # navy
        "Unknown":"#FFA500"   # orange
    }

    # 3) Apply Seaborn theme with this palette
    sns.set_theme(style="whitegrid", palette=SEX_COLORS)

    # 4) Set Matplotlib parameters
    mpl.rcParams.update({
        "figure.dpi": 100,
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
    })

def map_sex_series(series):
    """Map a pandas Series of raw sex values to standardized labels."""
    return series.map(lambda x: SEX_VALUE_MAP.get(x, 'Unknown'))




