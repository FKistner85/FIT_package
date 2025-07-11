# src/FIT_python/pipeline/feature_scaler_wrapper.py

from typing import Sequence
import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.preprocessing import StandardScaler, RobustScaler
from FIT_python.soft_config import SOFT_CONFIG


class FeatureScalerTransformer(TransformerMixin, BaseEstimator):
    """Scale numerical features using either a standard or robust approach."""

    def __init__(
        self,
        method: str = SOFT_CONFIG["general_pipeline_steps"]["scaler_default"]["method"],
        **scaler_kwargs,
    ):
        if method not in ("standard", "robust"):
            raise ValueError("method must be 'standard' or 'robust'")
        self.method = method
        self.scaler_kwargs = scaler_kwargs
        self.scaler = None
        self.feature_names_in_: Sequence[str] = []

    def fit(self, X, y=None):
        # determine whether input is DataFrame or ``ndarray``
        if isinstance(X, pd.DataFrame):
            arr = X.values
            self.feature_names_in_ = X.columns.to_list()
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
        # consistently convert to ``ndarray``
        if isinstance(X, pd.DataFrame):
            arr = X.values
        else:
            arr = np.asarray(X, dtype=float)

        arr_out = self.scaler.transform(arr)

        # return DataFrame with original labels if that was the input
        if isinstance(X, pd.DataFrame):
            return pd.DataFrame(arr_out, index=X.index, columns=self.feature_names_in_)
        # otherwise return ``ndarray``
        return arr_out


# Convenient presets for common scaler configurations
SCALER_PRESETS = {
    "standard": FeatureScalerTransformer(method="standard"),
    "robust": FeatureScalerTransformer(method="robust"),
}
