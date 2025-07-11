# src/FIT_python/pipeline/transform_wrapper.py

from typing import List
import pandas as pd
import numpy as np
from sklearn.base import TransformerMixin, BaseEstimator

from FIT_python.config import DEFAULT_TARGETS, OTTER_META_COLS


class NumericTransformer(TransformerMixin, BaseEstimator):
    """Convert feature columns to numeric ``numpy`` arrays.

    Steps
    -----
    1. Replace commas with dots in all feature columns.
    2. Coerce values to ``float`` (non-convertible values become ``NaN``).
    3. Return a plain ``numpy`` array ``X``.

    Target columns (``DEFAULT_TARGETS``) remain untouched and should be
    extracted outside of this transformer.
    """

    def __init__(self):
        self.feature_cols: List[str] = []

    def fit(self, df: pd.DataFrame, y=None):
        # determine meta columns: use ``OTTER_META_COLS`` if all are present, otherwise ``id`` only
        if all(col in df.columns for col in OTTER_META_COLS):
            meta = OTTER_META_COLS
        else:
            meta = ["id"]
        # treat all ``DEFAULT_TARGETS`` (e.g. ``['sex']``) as targets; the rest are features
        self.feature_cols = [
            col for col in df.columns if col not in meta + DEFAULT_TARGETS
        ]
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        # defensive copy
        df2 = df.copy()

        # 1) comma-to-dot replacement and string-to-number conversion
        for col in self.feature_cols:
            df2[col] = (
                df2[col]
                .astype(str)
                .str.replace(",", ".", regex=False)
                .pipe(pd.to_numeric, errors="coerce")
            )

        # 2) feature matrix as ``numpy`` array
        X = df2[self.feature_cols].to_numpy(dtype=float)
        return X
