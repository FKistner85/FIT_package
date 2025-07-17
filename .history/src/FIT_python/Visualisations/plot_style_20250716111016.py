import matplotlib as mpl
import seaborn as sns
from __future__ import annotations

# Mapping raw sex values to standardized labels
SEX_VALUE_MAP = {
    'f': 'Female', 'F': 'Female', 0: 'Female',
    'm': 'Male', 'M': 'Male', 1: 'Male'
}

# Sex-specific color palette
SEX_COLORS = {
    'Female': '#800000',  # dark red
    'Male':   '#000080',  # navy
    'Unknown':'#FFA500'   # orange
}

def map_sex(value):
    """Map raw sex values ('f','F',0,'m','M',1) to standardized labels."""
    return SEX_VALUE_MAP.get(value, 'Unknown')

def _lighten(color: str, amount: float) -> str:
    """Return a lighter shade of ``color``.

    ``amount`` specifies the blend ratio with white where ``0`` returns the
    original colour and ``1`` returns white.
    """
    r, g, b = mpl.colors.to_rgb(color)
    return mpl.colors.to_hex([
        r + (1 - r) * amount,
        g + (1 - g) * amount,
        b + (1 - b) * amount,
    ])

# Train/test color schemes
TRAIN_COLORS = SEX_COLORS
TEST_COLORS = {k: _lighten(v, 0.5) for k, v in SEX_COLORS.items()}

def apply_style() -> None:
    """Apply consistent plot styling across notebooks and modules,
    including sex-specific mappings and color palette.
    """
    # 1) Apply Seaborn theme with sex-specific palette
    sns.set_theme(style="whitegrid", palette=list(SEX_COLORS.values()))

    # 2) Update Matplotlib rc parameters for consistent styling
    mpl.rcParams.update({
        'figure.dpi': 100,
        'font.size': 12,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
    })
