# src/FIT_python/pipeline_sex/dimensionality_reduction_wrapper.py

import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, MDS, Isomap
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import umap

class DimensionalityReducerTransformer(TransformerMixin, BaseEstimator):
    """
    Wrapper für Dimensionsreduktion:
      - None: Identity (kein Reduzieren)
      - PCA (unsupervised)
      - UMAP (unsupervised + supervised)
      - t-SNE (unsupervised)
      - LDA (supervised)
      - MDS (unsupervised)
      - Isomap (unsupervised)
    """
    def __init__(
        self,
        method: str | None = None,
        n_components: int = 2,
        supervised: bool = False,
        **kwargs,
    ):
        """Initialize the reducer without altering passed parameters."""

        # store raw parameter values for sklearn cloning
        self.method = method
        self.n_components = n_components
        self.supervised = supervised
        self.kwargs = kwargs

        # sanitized values used internally
        method_norm = method.lower() if method is not None else None
        if method_norm not in (None, "pca", "umap", "tsne", "lda", "mds", "isomap"):
            raise ValueError(f"Unknown method: {method!r}")
        self._method_norm = method_norm
        self.requested_n = n_components

        self.reducer_ = None
        self.feature_names_out_: list[str] = []

    def get_params(self, deep=True):
        # Nur die Parameter aus __init__ zurückgeben, ohne interne kwargs
        return {
            "method": self.method,
            "n_components": self.n_components,
            "supervised": self.supervised,
        }

    def fit(self, X, y=None):
        # Identity-Fall
        if self._method_norm is None:
            if isinstance(X, pd.DataFrame):
                self.feature_names_out_ = X.columns.tolist()
            else:
                arr = np.asarray(X, dtype=float)
                self.feature_names_out_ = [f"f{i}" for i in range(arr.shape[1])]
            self.reducer_ = None
            return self

        arr = X.values if isinstance(X, pd.DataFrame) else np.asarray(X, dtype=float)
        n_samples, n_features = arr.shape
        max_c = min(n_samples, n_features)
        n_used = min(self.requested_n, max_c)

        if self._method_norm in ("umap", "lda") and self.supervised and y is None:
            raise ValueError(f"Supervised {self._method_norm.upper()} requires target labels `y`.")

        if self._method_norm == "pca":
            reducer = PCA(n_components=n_used, **self.kwargs).fit(arr)
        elif self._method_norm == "umap":
            if self.supervised and y is not None:
                reducer = umap.UMAP(n_components=n_used, target_metric="categorical", **self.kwargs).fit(arr, y)
            else:
                reducer = umap.UMAP(n_components=n_used, **self.kwargs).fit(arr)
        elif self._method_norm == "tsne":
            reducer = TSNE(n_components=n_used, **self.kwargs)
        elif self._method_norm == "lda":
            reducer = LinearDiscriminantAnalysis(n_components=n_used, **self.kwargs).fit(arr, y)
        elif self._method_norm == "mds":
            reducer = MDS(n_components=n_used, **self.kwargs).fit(arr)
        elif self._method_norm == "isomap":
            reducer = Isomap(n_components=n_used, **self.kwargs).fit(arr)
        else:
            raise ValueError(f"Unhandled reduction method: {self.method!r}")

        self.reducer_ = reducer
        self.feature_names_out_ = [f"{self._method_norm.upper()}{i+1}" for i in range(n_used)]
        return self

    def transform(self, X):
        # Identity-Fall: Daten unverändert zurück
        if self._method_norm is None:
            if isinstance(X, pd.DataFrame):
                return X.values
            return np.asarray(X, dtype=float)

        if self.reducer_ is None:
            raise RuntimeError("DimensionalityReducerTransformer must be fitted before transform.")

        arr = X.values if isinstance(X, pd.DataFrame) else np.asarray(X, dtype=float)
        if self._method_norm in ("pca", "umap", "lda", "mds", "isomap"):
            return self.reducer_.transform(arr)
        elif self._method_norm == "tsne":
            return self.reducer_.fit_transform(arr)

    def get_feature_names_out(self, input_features=None) -> list[str]:
        return self.feature_names_out_
