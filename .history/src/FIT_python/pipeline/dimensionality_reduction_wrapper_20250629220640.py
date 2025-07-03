# src/FIT_python/pipeline/dimensionality_reduction_wrapper.py

import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

try:
    import umap
except ImportError:
    umap = None


class DimensionalityReducerTransformer(TransformerMixin, BaseEstimator):
    """
    Wrapper für Dimensionsreduktion:
      - PCA
      - UMAP
      - t-SNE

    Passt n_components automatisch an, falls der gewünschte Wert
    größer ist als min(Anzahl Samples, Anzahl Features).
    """

    def __init__(
        self,
        method: str = "pca",
        n_components: int = 2,
        **kwargs
    ):
        method = method.lower()
        if method not in ("pca", "umap", "tsne"):
            raise ValueError(f"Unbekannte Methode: {method}")
        if method == "umap" and umap is None:
            raise ImportError("UMAP nicht installiert. Installiere 'umap-learn'.")
        self.method = method
        self.requested_n = n_components
        self.kwargs = kwargs
        self.reducer_ = None
        self.feature_names_out_: list[str] = []

    def fit(self, X, y=None):
        # Eingabe als ndarray
        arr = X.values if isinstance(X, pd.DataFrame) else np.asarray(X, dtype=float)
        n_samples, n_features = arr.shape

        # maximal mögliche Komponenten
        max_c = min(n_samples, n_features)
        n_used = min(self.requested_n, max_c)

        if self.method == "pca":
            self.reducer_ = PCA(n_components=n_used, **self.kwargs)
            self.reducer_.fit(arr)
        elif self.method == "umap":
            self.reducer_ = umap.UMAP(n_components=n_used, **self.kwargs)
            self.reducer_.fit(arr)
        else:  # tsne
            # sklearn's TSNE doesn't have a fit/transform API, so we fit in transform
            self.reducer_ = TSNE(n_components=n_used, **self.kwargs)

        # Namen für transformierte Features
        self.feature_names_out_ = [f"{self.method.upper()}{i+1}" for i in range(n_used)]
        return self

    def transform(self, X):
        if self.reducer_ is None:
            raise RuntimeError(
                "DimensionalityReducerTransformer muss zuerst gefittet werden."
            )
        arr = X.values if isinstance(X, pd.DataFrame) else np.asarray(X, dtype=float)
        if self.method == "tsne":
            # TSNE: direkt transform -> fit_transform
            return self.reducer_.fit_transform(arr)
        else:
            return self.reducer_.transform(arr)

    def get_feature_names_out(self, input_features=None) -> list[str]:
        return self.feature_names_out_
