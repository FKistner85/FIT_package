# src/FIT_python/pipeline/dimensionality_reduction_wrapper.py

import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap


class DimensionalityReducerTransformer(TransformerMixin, BaseEstimator):
    """
    Wrapper für Dimensionsreduktion:
      - PCA
      - UMAP
      - t-SNE (Barnes-Hut für n_components<=3, sonst Exact)
    """

    def __init__(self, method: str = "pca", n_components: int = 2, **kwargs):
        method = method.lower()
        if method not in ("pca", "umap", "tsne"):
            raise ValueError(f"Unknown method: {method!r}")
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

        # Initialisierung je nach Methode
        if self.method == "pca":
            reducer = PCA(n_components=n_used, **self.kwargs)
            reducer.fit(arr)

        elif self.method == "umap":
            reducer = umap.UMAP(n_components=n_used, **self.kwargs)
            reducer.fit(arr)

        else:  # tsne
            algo = "barnes_hut" if n_used <= 3 else "exact"
            reducer = TSNE(
                n_components=n_used,
                method=algo,
                **self.kwargs
            )
            # TSNE führt die Berechnung erst in fit_transform aus; wir speichern den
            # initialen Zustand, aber nicht fit(arr) aufrufen:
            pass

        self.reducer_ = reducer
        # Feature-Namen z.B. PCA1, UMAP1, TSNE1, ...
        prefix = self.method.upper()
        self.feature_names_out_ = [f"{prefix}{i+1}" for i in range(n_used)]
        return self

    def transform(self, X):
        if self.reducer_ is None:
            raise RuntimeError(
                "DimensionalityReducerTransformer must be fitted before transform."
            )
        arr = X.values if isinstance(X, pd.DataFrame) else np.asarray(X, dtype=float)

        if self.method == "pca":
            return self.reducer_.transform(arr)

        elif self.method == "umap":
            # UMAP unterstützt transform auf neue Daten
            return self.reducer_.transform(arr)

        else:  # tsne
            # TSNE hat keine separate transform-Methode – wir berechnen hier
            # eine neue Einbettung für die Eingabedaten.
            return self.reducer_.fit_transform(arr)

    def get_feature_names_out(self, input_features=None) -> list[str]:
        return self.feature_names_out_
