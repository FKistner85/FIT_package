# src/FIT_python/pipeline/transform_wrapper.py

from typing import List
import pandas as pd
import numpy as np
from sklearn.base import TransformerMixin, BaseEstimator

from FIT_python.config import DEFAULT_TARGETS, OTTER_META_COLS

class NumericTransformer(TransformerMixin, BaseEstimator):
    """
    Reiner Numeric-Transformer:
      1) Entfernt die Fold-Spalte (falls vorhanden)
      2) Komma → Punkt in allen Feature-Spalten
      3) Coercion zu float (non-convertible → NaN)
      4) Liefert reines NumPy-Array X zurück.
    
    Meta-Spalten sind entweder OTTER_META_COLS oder ['id'], 
    Ziel-Spalten DEFAULT_TARGETS werden nicht als Features behandelt.
    """

    def __init__(self):
        # Wird in fit() befüllt
        self.feature_cols: List[str] = []

    def fit(self, df: pd.DataFrame, y=None):
        # 1) Klonen und Fold-Spalte (Train/Test-Fold) entfernen
        df2 = df.drop(columns=['Fold'], errors='ignore')

        # 2) Meta-Spalten bestimmen
        if all(col in df2.columns for col in OTTER_META_COLS):
            meta_cols = OTTER_META_COLS
        else:
            meta_cols = ['id']

        # 3) Feature-Spalten: alles außer Meta + Ziel
        self.feature_cols = [
            col for col in df2.columns
            if col not in meta_cols + DEFAULT_TARGETS
        ]
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        # 1) Klonen und Fold-Spalte entfernen
        df2 = df.drop(columns=['Fold'], errors='ignore').copy()

        # 2) Komma→Punkt & String-to-Number in allen Features
        for col in self.feature_cols:
            df2[col] = (
                df2[col]
                  .astype(str)
                  .str.replace(',', '.', regex=False)
                  .pipe(pd.to_numeric, errors='coerce')
            )

        # 3) Feature-Matrix als NumPy-Array zurückgeben
        return df2[self.feature_cols].to_numpy(dtype=float)
