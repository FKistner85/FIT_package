"""Utilities for identifying and removing outliers."""

from __future__ import annotations

from pathlib import Path  # (falls extern genutzt)
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

from FIT_python.config import CONFIG
from FIT_python.utils import debug_report
from FIT_python.general_pipeline_steps.dimensionality_reduction_wrapper import (
    DimensionalityReducerTransformer,
)

# =============================================================================
# 1) Basic outlier cleaner (clip / z-score)
# =============================================================================


class OutlierCleanerTransformer(TransformerMixin, BaseEstimator):
    """Clean numerical outliers using clipping or z-score limiting.

    Methods
    -------
    'clip':
        clip all features to [q_low, q_high] (percentile clipping)
    'zscore':
        winsorise values outside ±z_thresh·σ to the boundary
    """

    def __init__(
        self,
        method: str = CONFIG["general_pipeline_steps"]["outlier_defaults"]["method"],
        lower_quantile: float = CONFIG["general_pipeline_steps"]["outlier_defaults"][
            "lower_quantile"
        ],
        upper_quantile: float = CONFIG["general_pipeline_steps"]["outlier_defaults"][
            "upper_quantile"
        ],
        z_thresh: float = CONFIG["general_pipeline_steps"]["outlier_defaults"]["z_thresh"],
    ):
        if method not in ("clip", "zscore"):
            raise ValueError("method must be 'clip' or 'zscore'")
        self.method = method
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile
        self.z_thresh = z_thresh
        self.bounds_: dict[str, np.ndarray] = {}
        self.numeric_cols_: list[str] | None = None

    def fit(self, X: pd.DataFrame | np.ndarray, y: Any = None) -> "OutlierCleanerTransformer":
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

    def transform(self, X: pd.DataFrame | np.ndarray) -> pd.DataFrame | np.ndarray:
        """Transform X by clipping or z-score limiting numeric columns."""
        if isinstance(X, pd.DataFrame):
            X_out = X.copy()
            cols = self.numeric_cols_ or X_out.select_dtypes(include="number").columns.tolist()
            arr = X_out[cols].values.astype(float)
        else:
            X_out = None
            cols = None
            arr = np.asarray(X, dtype=float)

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

        if X_out is not None:
            X_out[cols] = arr
            return X_out
        return arr


# =============================================================================
# 2) Centroid-based outlier marking on UMAP space
# =============================================================================


def mark_outliers_centroid(
    df: pd.DataFrame, *, bandwidth: float = 0.5, percentile: float = 1.0
) -> pd.DataFrame:
    """Mark outliers per individual in UMAP space using a Gaussian kernel.

    Expected columns: ['UMAP1', 'UMAP2', 'individual_id'].
    """
    from scipy.stats import multivariate_normal

    df = df.copy()
    df["centroid_density"] = np.nan
    df["is_outlier_centroid"] = False

    for _, idx in df.groupby("individual_id").groups.items():
        pts = df.loc[idx, ["UMAP1", "UMAP2"]].values
        cx, cy = pts.mean(axis=0)
        cov = [[bandwidth**2, 0], [0, bandwidth**2]]
        kernel = multivariate_normal(mean=[cx, cy], cov=cov)
        dens = kernel.pdf(pts)
        thresh = np.percentile(dens, percentile)

        df.loc[idx, "centroid_density"] = dens
        df.loc[idx, "is_outlier_centroid"] = dens < thresh

    return df


def print_filter_stats(name: str, df_before: pd.DataFrame, df_after: pd.DataFrame) -> None:
    """Print summary statistics about removed records."""
    n_before = len(df_before)
    n_after = len(df_after)
    removed = n_before - n_after
    pct_removed = 100 * removed / n_before if n_before else 0.0

    counts = df_after["individual_id"].value_counts()
    min_counts = counts.min() if not counts.empty else 0
    max_counts = counts.max() if not counts.empty else 0
    mean_counts = counts.mean() if not counts.empty else 0.0
    std_counts = counts.std() if not counts.empty else 0.0
    few_animals = int((counts < 3).sum()) if not counts.empty else 0

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

    Expects a DataFrame with columns ['UMAP1','UMAP2','individual_id'].
    """

    def __init__(self, *, bandwidth: float = 0.5, percentile: float = 1.0, drop: bool = False):
        self.bandwidth = bandwidth
        self.percentile = percentile
        self.drop = drop

    def fit(self, X: pd.DataFrame, y: Any = None) -> "CentroidOutlierTransformer":
        return self  # stateless

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise RuntimeError(
                "CentroidOutlierTransformer.transform expects a pandas.DataFrame "
                "with columns ['UMAP1','UMAP2','individual_id']."
            )
        df = mark_outliers_centroid(X, bandwidth=self.bandwidth, percentile=self.percentile)
        if self.drop:
            return df.loc[~df["is_outlier_centroid"]].drop(
                columns=["centroid_density", "is_outlier_centroid"]
            )
        return df


# =============================================================================
# 3) Wrapper: array/DataFrame → UMAP DF with IDs
# =============================================================================


class UMAPtoDF(TransformerMixin, BaseEstimator):
    """Convert features to a DataFrame with UMAP coordinates and IDs."""

    def __init__(self, **umap_kwargs: Any):
        # Preserve original behaviour: method='umap' with given kwargs
        self.reducer = DimensionalityReducerTransformer(method="umap", **umap_kwargs)

    def fit(self, X: pd.DataFrame, y: Any = None) -> "UMAPtoDF":
        # Must be a DataFrame including 'individual_id'
        self.reducer.fit(X.drop(columns=["individual_id"]), y)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        arr = self.reducer.transform(X.drop(columns=["individual_id"]))
        df = pd.DataFrame(arr, columns=["UMAP1", "UMAP2"], index=X.index)
        df["individual_id"] = X["individual_id"].values
        return df


# =============================================================================
# 4) Preset pipelines (kept behaviour-identical; DRY helpers)
# =============================================================================

def _umap(*, supervised: bool = True) -> UMAPtoDF:
    # In deinem Original stand supervised=True überall – das behalten wir bei.
    return UMAPtoDF(n_components=2, supervised=supervised)

def _centroid(percentile: float, *, bandwidth: float = 0.5, drop: bool = True) -> CentroidOutlierTransformer:
    return CentroidOutlierTransformer(bandwidth=bandwidth, percentile=percentile, drop=drop)

def _umap_centroid_pipeline(percentiles: list[float], *, supervised: bool = True) -> Pipeline:
    """
    Build a UMAP→Centroid pipeline; for iterative presets pass multiple percentiles.
    NOTE: Preserves the original behaviour (`supervised=True` even for "unsup"-labelled presets).
    """
    # Für 1-stufige Pipelines schöner benannt; für iterative nummeriert.
    steps: list[tuple[str, BaseEstimator]] = []
    if len(percentiles) == 1:
        steps = [("umap_df", _umap(supervised=supervised)), ("centroid", _centroid(percentiles[0]))]
    else:
        for i, p in enumerate(percentiles):
            steps.append((f"umap{i}", _umap(supervised=supervised)))
            steps.append((f"centroid{i}", _centroid(p)))
    return Pipeline(steps)


# Preset configurations combining common methods with sensible hyperparameters.
OUTLIER_PRESETS: dict[str, BaseEstimator] = {
    # Simple statistical clip/z-score
    "clip_90": OutlierCleanerTransformer(method="clip", lower_quantile=0.05, upper_quantile=0.95),
    "clip_98": OutlierCleanerTransformer(method="clip", lower_quantile=0.01, upper_quantile=0.99),
    "zscore_3": OutlierCleanerTransformer(method="zscore", z_thresh=3.0),

    # UMAP + centroid variants (supervised=True, wie im Original)
    "umap_centroid_1pct": _umap_centroid_pipeline([1.0], supervised=True),
    "sup_umap_centroid_1pct": _umap_centroid_pipeline([1.0], supervised=True),

    # Iterative pipelines
    "umap_centroid_iterative": _umap_centroid_pipeline([20.0, 5.0, 1.0], supervised=True),
    "sup_umap_centroid_iterative": _umap_centroid_pipeline([20.0, 5.0, 1.0], supervised=True),

    # „Hybrid“ – gleiche Sequenz; supervised=True beibehalten
    "hybrid_umap_centroid_iterative": _umap_centroid_pipeline([20.0, 5.0, 1.0], supervised=True),
}
