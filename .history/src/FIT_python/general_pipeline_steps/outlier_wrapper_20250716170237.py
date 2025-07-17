# src/FIT_python/pipeline/outlier_wrapper.py

import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin, BaseEstimator
from FIT_python.soft_config import SOFT_CONFIG


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
        method: str = SOFT_CONFIG["general_pipeline_steps"]["outlier_defaults"][
            "method"
        ],
        lower_quantile: float = SOFT_CONFIG["general_pipeline_steps"][
            "outlier_defaults"
        ]["lower_quantile"],
        upper_quantile: float = SOFT_CONFIG["general_pipeline_steps"][
            "outlier_defaults"
        ]["upper_quantile"],
        z_thresh: float = SOFT_CONFIG["general_pipeline_steps"]["outlier_defaults"][
            "z_thresh"
        ],
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


# --- 1) Centroid‑basierte Outlier‑Markierung pro Gruppe ---
def mark_outliers_centroid(df, bandwidth=0.5, percentile=1):
    """
    Berechnet für jede individual_id separat:
      1. Den Centroid der UMAP-Punkte.
      2. Eine Gauß‑KDE um diesen Centroid (Varianz = bandwidth^2).
      3. Die Dichte jedes echten Punktes unter diesem Kernel.
      4. Markiert als Outlier alle Punkte unterhalb des gegebenen Perzentils.
    """
    df = df.copy()
    df['centroid_density']    = np.nan
    df['is_outlier_centroid'] = False
    
    for ind, idx in df.groupby('individual_id').groups.items():
        pts = df.loc[idx, ['UMAP1','UMAP2']].values
        # 1) Centroid
        cx, cy = pts.mean(axis=0)
        # 2) Kernel um Centroid
        cov    = [[bandwidth**2, 0], [0, bandwidth**2]]
        kernel = multivariate_normal(mean=[cx, cy], cov=cov)
        # 3) Dichte-Bewertung
        dens   = kernel.pdf(pts)
        # 4) Schwellenwert am Perzentil
        thresh = np.percentile(dens, percentile)
        
        df.loc[idx, 'centroid_density']    = dens
        df.loc[idx, 'is_outlier_centroid'] = dens < thresh
        
    return df