"""Utilities for identifying and removing outliers."""

# src/FIT_python/pipeline/outlier_wrapper.py

import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.pipeline import Pipeline
from FIT_python.soft_config import SOFT_CONFIG
from FIT_python.utils import debug_report
from FIT_python.general_pipeline_steps.dimensionality_reduction_wrapper import (
    DimensionalityReducerTransformer,
)


class OutlierCleanerTransformer(TransformerMixin, BaseEstimator):
    """Clean numerical outliers using clipping or z-score limiting.

    Methods
    -------
    - ``'clip'``: clip all features to ``[q_low, q_high]`` (percentile clipping)
    - ``'zscore'``: set values outside ``±z_thresh*σ`` to the boundary (Winsorizing)
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
        """Store configuration for clipping or z-score cleaning.

        Parameters
        ----------
        method:
            Cleaning algorithm. Options: ``'clip'`` or ``'zscore'``.
            Defaults to the value from :data:`SOFT_CONFIG`.
        lower_quantile, upper_quantile:
            Bounds used when ``method='clip'``. Defaults come from
            :data:`SOFT_CONFIG`.
        z_thresh:
            Z-score threshold when ``method='zscore'``. Default from
            :data:`SOFT_CONFIG`.
        """
        if method not in ("clip", "zscore"):
            raise ValueError("method must be 'clip' or 'zscore'")
        self.method = method
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile
        self.z_thresh = z_thresh
        self.bounds_ = {}

    def fit(self, X, y=None):
        """Calculate clipping or z-score bounds on numeric columns only."""
        if isinstance(X, pd.DataFrame):
            X_num = X.select_dtypes(include="number")
            self.numeric_cols_ = X_num.columns.to_list()
            arr = X_num.values.astype(float)
        else:
            self.numeric_cols_ = None
            arr = np.asarray(X, dtype=float)
        if self.method == "clip":
            lows = np.quantile(arr, self.lower_quantile, axis=0)
            highs = np.quantile(arr, self.upper_quantile, axis=0)
            self.bounds_ = {"low": lows, "high": highs}
        else:
            means = np.mean(arr, axis=0)
            stds = np.std(arr, axis=0)
            self.bounds_ = {"mean": means, "std": stds}
        return self

    def transform(self, X):
        """Transform ``X`` by clipping or z-score limiting numeric columns."""
        if isinstance(X, pd.DataFrame):
            X_out = X.copy()
            cols = self.numeric_cols_ or X_out.select_dtypes(include="number").columns.tolist()
            arr = X_out[cols].values.astype(float)
        else:
            arr = np.asarray(X, dtype=float)
            cols = None
        if self.method == "clip":
            low, high = self.bounds_["low"], self.bounds_["high"]
            arr = np.minimum(np.maximum(arr, low), high)
        else:
            mean, std = self.bounds_["mean"], self.bounds_["std"]
            zt = self.z_thresh
            lower = mean - zt * std
            upper = mean + zt * std
            arr = np.minimum(np.maximum(arr, lower), upper)
        debug_report(arr, "outlier_clean")
        if isinstance(X, pd.DataFrame):
            X_out[cols] = arr
            return X_out
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
    from scipy.stats import multivariate_normal

    df = df.copy()
    df["centroid_density"] = np.nan
    df["is_outlier_centroid"] = False

    for ind, idx in df.groupby("individual_id").groups.items():
        pts = df.loc[idx, ["UMAP1", "UMAP2"]].values
        cx, cy = pts.mean(axis=0)
        cov = [[bandwidth**2, 0], [0, bandwidth**2]]
        kernel = multivariate_normal(mean=[cx, cy], cov=cov)
        dens = kernel.pdf(pts)
        thresh = np.percentile(dens, percentile)

        df.loc[idx, "centroid_density"] = dens
        df.loc[idx, "is_outlier_centroid"] = dens < thresh

    return df


def print_filter_stats(name, df_before, df_after):
    """Print summary statistics about removed records."""
    n_before = len(df_before)
    n_after = len(df_after)
    removed = n_before - n_after
    pct_removed = 100 * removed / n_before

    counts = df_after["individual_id"].value_counts()
    min_counts = counts.min()
    max_counts = counts.max()
    mean_counts = counts.mean()
    std_counts = counts.std()
    few_animals = (counts < 3).sum()

    print(f"{name}")
    print(f"{'Punkte vorher:':25}{n_before}")
    print(f"{'Punkte danach:':25}{n_after}")
    print(f"{'Entfernt:':25}{removed} ({pct_removed:.1f}%)")
    print(f"{'Min Beobacht./ID:':25}{min_counts}")
    print(f"{'Max Beobacht./ID:':25}{max_counts}")
    print(f"{'Ø Beobacht./ID:':25}{mean_counts:.2f} ± {std_counts:.2f}")
    print(f"{'<3 Beobachtungen Tiere:':25}{few_animals}\n")


class CentroidOutlierTransformer(TransformerMixin, BaseEstimator):
    """Mark or remove outliers per individual in UMAP space.
    Expects a DataFrame with columns ['UMAP1','UMAP2','individual_id']."""

    def __init__(
        self, bandwidth: float = 0.5, percentile: float = 1.0, drop: bool = False
    ):
        """Store configuration for clipping or z-score cleaning."""
        """Store parameters."""
        self.bandwidth = bandwidth
        self.percentile = percentile
        self.drop = drop

    def fit(self, X, y=None):
        """No fitting necessary."""
        return self

    def transform(self, X):
        """Transform the DataFrame by clipping or z-score limiting."""
        if not isinstance(X, pd.DataFrame):
            raise RuntimeError(
                "CentroidOutlierTransformer.transform erwartet einen pandas.DataFrame "
                "mit Spalten ['UMAP1','UMAP2','individual_id']."
            )
        df = mark_outliers_centroid(
            X, bandwidth=self.bandwidth, percentile=self.percentile
        )
        if self.drop:
            return df.loc[~df["is_outlier_centroid"]].drop(
                columns=["centroid_density", "is_outlier_centroid"]
            )
        return df


# --- 2) Wrapper, der UMAP → DataFrame konvertiert ---


class UMAPtoDF(TransformerMixin, BaseEstimator):
    """Convert arrays to a DataFrame with UMAP coordinates and IDs."""

    def __init__(self, **umap_kwargs):
        """Store configuration for clipping or z-score cleaning."""
        """Store UMAP reducer parameters."""
        self.reducer = DimensionalityReducerTransformer(method="umap", **umap_kwargs)

    def fit(self, X, y=None):
        """No fitting necessary."""
        # X must be DataFrame including 'individual_id'
        self.reducer.fit(X.drop(columns=["individual_id"]), y)
        return self

    def transform(self, X):
        """Return a DataFrame with UMAP coordinates and IDs."""
        arr = self.reducer.transform(X.drop(columns=["individual_id"]))
        df = pd.DataFrame(arr, columns=["UMAP1", "UMAP2"], index=X.index)
        df["individual_id"] = X["individual_id"].values
        return df


# --- 3) Pipeline‑Presets mit UMAPtoDF + CentroidOutlierTransformer ---

OUTLIER_PRESETS = {}

# 3.1 Unsupervised UMAP → DF → 1% drop
OUTLIER_PRESETS["umap_centroid_1pct"] = Pipeline(
    [
        ("umap_df", UMAPtoDF(n_components=2)),
        (
            "centroid",
            CentroidOutlierTransformer(bandwidth=0.5, percentile=1.0, drop=True),
        ),
    ]
)

# 3.2 Supervised UMAP → DF → 1% drop
OUTLIER_PRESETS["sup_umap_centroid_1pct"] = Pipeline(
    [
        ("umap_df", UMAPtoDF(n_components=2, supervised=True)),
        (
            "centroid",
            CentroidOutlierTransformer(bandwidth=0.5, percentile=1.0, drop=True),
        ),
    ]
)

# 3.3 Iterative unsupervised: 20%→UMAP→5%→UMAP→1%
OUTLIER_PRESETS["umap_centroid_iterative"] = Pipeline(
    [
        ("umap0", UMAPtoDF(n_components=2)),
        (
            "coarse20",
            CentroidOutlierTransformer(bandwidth=0.5, percentile=20.0, drop=True),
        ),
        ("umap1", UMAPtoDF(n_components=2)),
        ("fine5", CentroidOutlierTransformer(bandwidth=0.5, percentile=5.0, drop=True)),
        ("umap2", UMAPtoDF(n_components=2)),
        ("very1", CentroidOutlierTransformer(bandwidth=0.5, percentile=1.0, drop=True)),
    ]
)

# 3.4 Iterative supervised: 20%→sup-UMAP→5%→sup-UMAP→1%
OUTLIER_PRESETS["sup_umap_centroid_iterative"] = Pipeline(
    [
        ("umap0", UMAPtoDF(n_components=2, supervised=True)),
        (
            "coarse20",
            CentroidOutlierTransformer(bandwidth=0.5, percentile=20.0, drop=True),
        ),
        ("umap1", UMAPtoDF(n_components=2, supervised=True)),
        ("fine5", CentroidOutlierTransformer(bandwidth=0.5, percentile=5.0, drop=True)),
        ("umap2", UMAPtoDF(n_components=2, supervised=True)),
        ("very1", CentroidOutlierTransformer(bandwidth=0.5, percentile=1.0, drop=True)),
    ]
)

# 3.5 Hybrid unsup→sup→unsup
OUTLIER_PRESETS["hybrid_umap_centroid_iterative"] = Pipeline(
    [
        ("umap0", UMAPtoDF(n_components=2, supervised=False)),
        (
            "coarse20",
            CentroidOutlierTransformer(bandwidth=0.5, percentile=20.0, drop=True),
        ),
        ("umap1", UMAPtoDF(n_components=2, supervised=True)),
        ("fine5", CentroidOutlierTransformer(bandwidth=0.5, percentile=5.0, drop=True)),
        ("umap2", UMAPtoDF(n_components=2, supervised=False)),
        ("very1", CentroidOutlierTransformer(bandwidth=0.5, percentile=1.0, drop=True)),
    ]
)
