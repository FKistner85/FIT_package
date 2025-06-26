# src/FIT_python/pipeline/feature_scaler_wrapper.py

from typing import Sequence
import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.preprocessing import StandardScaler, RobustScaler

class FeatureScalerTransformer(TransformerMixin, BaseEstimator):
    """
    Scaler für numerische Features, mit zwei Modi:
      - method='standard': StandardScaler (z-Transformation)
      - method='robust':   RobustScaler (Median & IQR)
    """

    def __init__(self, method: str = "standard", **scaler_kwargs):
        if method not in ("standard", "robust"):
            raise ValueError("method must be 'standard' or 'robust'")
        self.method = method
        self.scaler_kwargs = scaler_kwargs
        self.scaler = None
        self.feature_names_in_: Sequence[str] = []

    def fit(self, X, y=None):
        # Ermitteln, ob DataFrame oder ndarray
        if isinstance(X, pd.DataFrame):
            arr = X.values
            self.feature_names_in_ = X.columns.to_list()
        else:
            arr = np.asarray(X, dtype=float)
            # wenn ndarray, setzen wir Dummy-Spaltennamen f0, f1, ...
            self.feature_names_in_ = [f"f{i}" for i in range(arr.shape[1])]

        # Scaler initialisieren
        if self.method == "standard":
            self.scaler = StandardScaler(**self.scaler_kwargs)
        else:
            self.scaler = RobustScaler(**self.scaler_kwargs)

        # Fit auf alle Spalten
        self.scaler.fit(arr)
        return self

    def transform(self, X):
        # Wandle einheitlich in ndarray um
        if isinstance(X, pd.DataFrame):
            arr = X.values
        else:
            arr = np.asarray(X, dtype=float)

        arr_out = self.scaler.transform(arr)

        # Falls DataFrame reinkam, gib DataFrame gleichen Labels zurück
        if isinstance(X, pd.DataFrame):
            return pd.DataFrame(arr_out, index=X.index, columns=self.feature_names_in_)
        # sonst reines ndarray
        return arr_out
