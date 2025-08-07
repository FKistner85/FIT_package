# src/FIT_python/pipeline/feature_scaler_wrapper.py

from typing import Sequence
import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.preprocessing import StandardScaler, RobustScaler
from FIT_python.config import CONFIG
from FIT_python.utils import debug_report


class FeatureScalerTransformer(TransformerMixin, BaseEstimator):
    """Scale numerical features using either a standard or robust approach."""

    def __init__(
        self,
        method: str = CONFIG["general_pipeline_steps"]["scaler_default"]["method"],
        **scaler_kwargs,
    ):
        """Create the transformer.

        Parameters
        ----------
        method:
            Scaling strategy. Options: ``'standard'`` or ``'robust'``.
            Defaults to the value from :data:`CONFIG`.
        scaler_kwargs:
            Additional arguments passed to the underlying scaler.
        """
        if method not in ("standard", "robust"):
            raise ValueError("method must be 'standard' or 'robust'")
        self.method = method
        self.scaler_kwargs = scaler_kwargs
        self.scaler = None
        self.feature_names_in_: Sequence[str] = []

    def fit(self, X, y=None):
        # determine whether input is DataFrame or ``ndarray``
        if isinstance(X, pd.DataFrame):
            X_num = X.select_dtypes(include="number")
            arr = X_num.to_numpy(dtype=float)
            self.feature_names_in_ = X_num.columns.to_list()
        else:
            arr = np.asarray(X, dtype=float)
            # assign dummy column names ``f0``, ``f1``, ... for ``ndarray`` input
            self.feature_names_in_ = [f"f{i}" for i in range(arr.shape[1])]

        # initialise the scaler
        if self.method == "standard":
            self.scaler = StandardScaler(**self.scaler_kwargs)
        else:
            self.scaler = RobustScaler(**self.scaler_kwargs)

        # fit on all columns
        self.scaler.fit(arr)
        return self

    def transform(self, X):
        # consistently convert to ``ndarray`` and only scale numeric columns
        if isinstance(X, pd.DataFrame):
            X_out = X.copy()
            arr = X_out[self.feature_names_in_].to_numpy(dtype=float)
        else:
            arr = np.asarray(X, dtype=float)
            X_out = None

        arr_out = self.scaler.transform(arr)
        debug_report(arr_out, "scale")

        if isinstance(X, pd.DataFrame):
            X_out[self.feature_names_in_] = arr_out
            return X_out
        return arr_out


# Convenient presets for common scaler configurations
SCALER_PRESETS = {
    "standard": FeatureScalerTransformer(method="standard"),
    "robust": FeatureScalerTransformer(method="robust"),
}
