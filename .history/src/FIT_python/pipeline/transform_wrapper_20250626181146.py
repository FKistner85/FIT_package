# src/FIT_python/pipeline/transform_wrapper.py

from typing import List
import pandas as pd
import numpy as np
from sklearn.base import TransformerMixin, BaseEstimator

from FIT_python.config import DEFAULT_TARGETS, OTTER_META_COLS

class NumericTransformer(TransformerMixin, BaseEstimator):
    """
    Reiner Numeric-Transformer:
      1) Komma → Punkt in allen Feature-Spalten
      2) Coercion zu float (non-convertible → NaN)
      3) Liefert reines NumPy-Array X zurück.
    
    Die Zielspalten (DEFAULT_TARGETS) bleiben unberührt und sollten
    außerhalb dieses Transformers separat extrahiert werden.
    """

    def __init__(self):
        self.feature_cols: List[str] = []

    def fit(self, df: pd.DataFrame, y=None):
        # Bestimme Meta-Spalten: falls OTTER_META_COLS komplett enthalten, sonst nur 'id'
        if all(col in df.columns for col in OTTER_META_COLS):
            meta = OTTER_META_COLS
        else:
            meta = ['id']
        # Alle DEFAULT_TARGETS (z.B. ['sex']) als Ziel, Rest sind Features
        self.feature_cols = [
            col for col in df.columns
            if col not in meta + DEFAULT_TARGETS
        ]
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        # Defensive Copy
        df2 = df.copy()

        # 1) Komma→Punkt & String-to-Number
        for col in self.feature_cols:
            df2[col] = (
                df2[col]
                  .astype(str)
                  .str.replace(',', '.', regex=False)
                  .pipe(pd.to_numeric, errors='coerce')
            )

        # 2) Feature-Matrix als NumPy
        X = df2[self.feature_cols].to_numpy(dtype=float)
        return X
