# src/FIT_python/pipeline/dimensionality_reduction_wrapper.py

import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.decomposition import PCA

class DimensionalityReducerTransformer(TransformerMixin, BaseEstimator):
    """
    Wrapper für Dimensionsreduktion (aktuell nur PCA).
    Passt n_components automatisch an, falls der gewünschte Wert
    größer ist als min(Anzahl Samples, Anzahl Features).
    """

    def __init__(self, method: str = "pca", n_components: int = 2, **kwargs):
        if method.lower() != "pca":
            raise ValueError(f"Unbekannte Methode: {method}")
        self.method = method.lower()
        self.requested_n = n_components
        self.kwargs = kwargs
        self.reducer_: PCA | None = None
        self.feature_names_out_: list[str] = []

    def fit(self, X, y=None):
        # Eingabe als ndarray
        arr = X.values if isinstance(X, pd.DataFrame) else np.asarray(X, dtype=float)
        n_samples, n_features = arr.shape

        # maximal mögliche Komponenten
        max_c = min(n_samples, n_features)
        if self.requested_n > max_c:
            # Fallback
            n_used = max_c
        else:
            n_used = self.requested_n

        # PCA initialisieren und fitten
        self.reducer_ = PCA(n_components=n_used, **self.kwargs)
        self.reducer_.fit(arr)

        # Namen für transformierte Features
        self.feature_names_out_ = [f"PC{i+1}" for i in range(n_used)]
        return self

    def transform(self, X):
        if self.reducer_ is None:
            raise RuntimeError("DimensionalityReducerTransformer muss zuerst gefittet werden.")
        arr = X.values if isinstance(X, pd.DataFrame) else np.asarray(X, dtype=float)
        return self.reducer_.transform(arr)

    def get_feature_names_out(self, input_features=None) -> list[str]:
        return self.feature_names_out_
