# src/FIT_python/pipeline/outlier_wrapper.py

import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin, BaseEstimator


class OutlierCleanerTransformer(TransformerMixin, BaseEstimator):
    """Clean numerical outliers using clipping or z-score limiting.

    Methods
    -------
    - ``'clip'``: clip all features to ``[q_low, q_high]`` (percentile clipping)
    - ``'zscore'``: set values outside ``±z_thresh*σ`` to the boundary (Winsorizing)

    Parameters
    ----------
    method : str
        ``'clip'`` or ``'zscore'``
    lower_quantile, upper_quantile : float
        Quantiles used for clipping mode, e.g. ``0.01`` and ``0.99``
    z_thresh : float
        Z-score threshold for ``'zscore'`` mode, e.g. ``3.0``
    """

    def __init__(
        self,
        method: str = "clip",
        lower_quantile: float = 0.01,
        upper_quantile: float = 0.99,
        z_thresh: float = 3.0,
    ):
        if method not in ("clip", "zscore"):
            raise ValueError("method must be 'clip' or 'zscore'")
        self.method = method
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile
        self.z_thresh = z_thresh
        self.bounds_ = {}

    def fit(self, X, y=None):
        # X may be a DataFrame or ``ndarray``
        if isinstance(X, pd.DataFrame):
            arr = X.values
            cols = X.columns
        else:
            arr = np.asarray(X, dtype=float)
            cols = [f"f{i}" for i in range(arr.shape[1])]

        if self.method == "clip":
            # determine quantiles per column
            lows = np.quantile(arr, self.lower_quantile, axis=0)
            highs = np.quantile(arr, self.upper_quantile, axis=0)
            self.bounds_ = {"low": lows, "high": highs, "cols": cols}
        else:
            # determine z-score bounds
            means = np.mean(arr, axis=0)
            stds = np.std(arr, axis=0)
            self.bounds_ = {"mean": means, "std": stds, "cols": cols}

        return self

    def transform(self, X):
        # convert to ndarray
        if isinstance(X, pd.DataFrame):
            arr = X.values.copy()
            cols = X.columns
        else:
            arr = np.asarray(X, dtype=float)
            cols = self.bounds_.get("cols", [f"f{i}" for i in range(arr.shape[1])])

        if self.method == "clip":
            low = self.bounds_["low"]
            high = self.bounds_["high"]
            arr = np.minimum(np.maximum(arr, low), high)
        else:
            mean = self.bounds_["mean"]
            std = self.bounds_["std"]
            zt = self.z_thresh
            lower = mean - zt * std
            upper = mean + zt * std
            arr = np.minimum(np.maximum(arr, lower), upper)

        # return DataFrame if we received one
        if isinstance(X, pd.DataFrame):
            return pd.DataFrame(arr, index=X.index, columns=cols)
        return arr


# Preset configurations combining common methods with sensible hyperparameters.
OUTLIER_PRESETS = {
    "clip_90": OutlierCleanerTransformer(
        method="clip", lower_quantile=0.05, upper_quantile=0.95
    ),
    "clip_98": OutlierCleanerTransformer(
        method="clip", lower_quantile=0.01, upper_quantile=0.99
    ),
    "zscore_3": OutlierCleanerTransformer(method="zscore", z_thresh=3.0),
}


def plot_feature_distributions(
    df_before: pd.DataFrame, df_after: pd.DataFrame, out_dir: "Path | str"
) -> None:
    """Save side-by-side histograms for numeric columns before and after cleaning.

    Parameters
    ----------
    df_before : pd.DataFrame
        Raw input dataframe prior to cleaning.
    df_after : pd.DataFrame
        Dataframe after cleaning/outlier removal.
    out_dir : Path | str
        Directory where PNG/SVG figures will be written.
    """
    from pathlib import Path
    import matplotlib.pyplot as plt
    from FIT_python.plot_style import apply_style

    apply_style()

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    num_cols = df_before.select_dtypes(include=[np.number]).columns.intersection(
        df_after.select_dtypes(include=[np.number]).columns
    )[:30]

    for col in num_cols:
        fig, axes = plt.subplots(1, 2, figsize=(8, 3))
        axes[0].hist(df_before[col].dropna(), bins=30, color="grey", edgecolor="black")
        axes[0].set_xlabel(col)
        axes[1].hist(
            df_after[col].dropna(), bins=30, color="steelblue", edgecolor="black"
        )
        axes[1].set_xlabel(col)
        from FIT_python.caption_utils import save_caption

        fig.tight_layout()
        caption = f"Distribution of {col} before and after cleaning"
        for ext in ("png", "svg"):
            file = out_path / f"{col}.{ext}"
            fig.savefig(file)
            save_caption(file, caption)
        plt.close(fig)
