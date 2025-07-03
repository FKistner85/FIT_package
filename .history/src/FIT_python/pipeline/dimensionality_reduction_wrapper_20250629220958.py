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
      - method="pca"  → sklearn.decomposition.PCA
      - method="umap" → umap.UMAP
      - method="tsne" → sklearn.manifold.TSNE
    Passt n_components automatisch an, falls angefragt > min(n_samples,n_features).
    """
    def __init__(self, method: str = "pca", n_components: int = 2, **kwargs):
        method = method.lower()
        if method not in ("pca", "umap", "tsne"):
            raise ValueError(f"Unbekannte Methode: {method}")
        self.method = method
        self.requested_n = n_components
        self.kwargs = kwargs
        self.reducer_ = None
        self.feature_names_out_ = []

    def fit(self, X, y=None):
        arr = X.values if isinstance(X, pd.DataFrame) else np.asarray(X, float)
        n_samples, n_features = arr.shape
        max_c = min(n_samples, n_features)
        n_used = min(self.requested_n, max_c)

        if self.method == "pca":
            self.reducer_ = PCA(n_components=n_used, **self.kwargs)
        elif self.method == "umap":
            self.reducer_ = umap.UMAP(n_components=n_used, **self.kwargs)
        else:  # tsne
            # note: TSNE ignores random_state if not passed here
            self.reducer_ = TSNE(n_components=n_used, **self.kwargs)

        self.reducer_.fit(arr)
        self.feature_names_out_ = [f"{self.method.upper()}{i+1}" for i in range(n_used)]
        return self

    def transform(self, X):
        if self.reducer_ is None:
            raise RuntimeError("DimensionalityReducerTransformer muss zuerst gefittet werden.")
        arr = X.values if isinstance(X, pd.DataFrame) else np.asarray(X, float)
        return self.reducer_.transform(arr)

    def get_feature_names_out(self, input_features=None) -> list[str]:
        return self.feature_names_out_
