# src/FIT_python/pipeline/outlier_wrapper.py

import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin, BaseEstimator

class OutlierCleanerTransformer(TransformerMixin, BaseEstimator):
    """
    Outlier-Bereinigung durch Clipping oder Z-Score-Begrenzung.
    
    Methoden:
      - 'clip': alle Features auf [q_low, q_high] clippen (Percentile-Clipping)
      - 'zscore': Werte außerhalb von ±z_thresh*σ auf ±z_thresh*σ setzten (Winsorizing)
      
    Parameter:
      - method: 'clip' oder 'zscore'
      - lower_quantile, upper_quantile: für 'clip' (z.B. 0.01, 0.99)
      - z_thresh: für 'zscore' (z.B. 3.0)
    """
    def __init__(
        self,
        method: str = "clip",
        lower_quantile: float = 0.01,
        upper_quantile: float = 0.99,
        z_thresh: float = 3.0
    ):
        if method not in ("clip", "zscore"):
            raise ValueError("method must be 'clip' or 'zscore'")
        self.method = method
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile
        self.z_thresh = z_thresh
        self.bounds_ = {}

    def fit(self, X, y=None):
        # X kann DataFrame oder ndarray sein
        if isinstance(X, pd.DataFrame):
            arr = X.values
            cols = X.columns
        else:
            arr = np.asarray(X, dtype=float)
            cols = [f"f{i}" for i in range(arr.shape[1])]

        if self.method == "clip":
            # Per-Spalte Quantile bestimmen
            lows = np.quantile(arr, self.lower_quantile, axis=0)
            highs = np.quantile(arr, self.upper_quantile, axis=0)
            self.bounds_ = {"low": lows, "high": highs, "cols": cols}
        else:
            # Z-Score-Grenzen bestimmen
            means = np.mean(arr, axis=0)
            stds  = np.std(arr, axis=0)
            self.bounds_ = {"mean": means, "std": stds, "cols": cols}

        return self

    def transform(self, X):
        # in ndarray überführen
        if isinstance(X, pd.DataFrame):
            arr = X.values.copy()
            cols = X.columns
        else:
            arr = np.asarray(X, dtype=float)
            cols = self.bounds_.get("cols", [f"f{i}" for i in range(arr.shape[1])])

        if self.method == "clip":
            low  = self.bounds_["low"]
            high = self.bounds_["high"]
            arr = np.minimum(np.maximum(arr, low), high)
        else:
            mean = self.bounds_["mean"]
            std  = self.bounds_["std"]
            zt   = self.z_thresh
            lower = mean - zt * std
            upper = mean + zt * std
            arr = np.minimum(np.maximum(arr, lower), upper)

        # Wenn DataFrame reinkam, gib DataFrame zurück
        if isinstance(X, pd.DataFrame):
            return pd.DataFrame(arr, index=X.index, columns=cols)
        return arr
