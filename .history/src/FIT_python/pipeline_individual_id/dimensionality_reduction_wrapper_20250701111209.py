# src/FIT_python/pipeline/dimensionality_reduction_wrapper.py

import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


class DimensionalityReducerTransformer(TransformerMixin, BaseEstimator):
    """
    Wrapper für Dimensionsreduktion:
      - PCA (unsupervised)
      - UMAP (unsupervised + supervised)
      - t-SNE (unsupervised)
      - LDA (supervised)
    """

    def __init__(self, method: str = "pca", n_components: int = 2, supervised: bool = False, **kwargs):
        method = method.lower()
        if method not in ("pca", "umap", "tsne", "lda"):
            raise ValueError(f"Unknown method: {method!r}")
        self.method = method
        self.requested_n = n_components
        self.supervised = supervised
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

        # Optional prüfen ob supervised Mode y benötigt
        if self.supervised and y is None:
            raise ValueError(f"Supervised {self.method.upper()} requires target labels `y`.")

        # Initialisierung je nach Methode
        if self.method == "pca":
            reducer = PCA(n_components=n_used, **self.kwargs)
            reducer.fit(arr)

        elif self.method == "umap":
            if self.supervised and y is not None:
                reducer = umap.UMAP(n_components=n_used, target_metric='categorical', **self.kwargs)
                reducer.fit(arr, y)
            else:
                reducer = umap.UMAP(n_components=n_used, **self.kwargs)
                reducer.fit(arr)

        elif self.method == "lda":
            if y is None:
                raise ValueError("LDA requires class labels (y)")
            reducer = LinearDiscriminantAnalysis(n_components=n_used, **self.kwargs)
            reducer.fit(arr, y)

        else:  # tsne
            algo = "barnes_hut" if n_used <= 3 else "exact"
            reducer = TSNE(
                n_components=n_used,
                method=algo,
                **self.kwargs
            )
            # TSNE führt die Berechnung erst in fit_transform aus
            pass

        self.reducer_ = reducer
        prefix = self.method.upper()
        self.feature_names_out_ = [f"{prefix}{i+1}" for i in range(n_used)]
        return self

    def transform(self, X):
        if self.reducer_ is None:
            raise RuntimeError(
                "DimensionalityReducerTransformer must be fitted before transform."
            )
        arr = X.values if isinstance(X, pd.DataFrame) else np.asarray(X, dtype=float)

        if self.method in ("pca", "umap", "lda"):
            return self.reducer_.transform(arr)

        else:  # tsne
            return self.reducer_.fit_transform(arr)

    def get_feature_names_out(self, input_features=None) -> list[str]:
        return self.feature_names_out_
