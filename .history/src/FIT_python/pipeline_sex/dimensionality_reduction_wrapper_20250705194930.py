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
      - None: Identity
      - PCA
      - UMAP
      - t-SNE
      - LDA
      - MDS
      - Isomap
    """
    def __init__(
        self,
        method: str | None = None,
        n_components: int = 2,
        supervised: bool = False,
        # explizite UMAP-Parameter
        n_neighbors: int = 15,
        min_dist: float = 0.1,
        # optional: PCA-Paramter
        whiten: bool = False,
        **unused
    ):
        self.method       = method
        self.n_components = n_components
        self.supervised   = supervised
        # UMAP
        self.n_neighbors = n_neighbors
        self.min_dist    = min_dist
        # PCA
        self.whiten      = whiten

        self._method_norm = method.lower() if method is not None else None
        if self._method_norm not in (None, "pca","umap","tsne","lda","mds","isomap"):
            raise ValueError(f"Unknown method: {method!r}")

        self.reducer_ = None
        self.feature_names_out_: list[str] = []

    def get_params(self, deep=True):
        return {
            "method":       self.method,
            "n_components": self.n_components,
            "supervised":   self.supervised,
            "n_neighbors":  self.n_neighbors,
            "min_dist":     self.min_dist,
            "whiten":       self.whiten,
        }

    def fit(self, X, y=None):
        # Identity
        if self._method_norm is None:
            df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
            self.feature_names_out_ = df.columns.tolist()
            self.reducer_ = None
            return self

        arr = (X.values if isinstance(X, pd.DataFrame) else np.asarray(X, float))
        n_used = min(self.n_components, arr.shape[0], arr.shape[1])

        if self._method_norm == "pca":
            self.reducer_ = PCA(n_components=n_used, whiten=self.whiten).fit(arr)
        elif self._method_norm == "umap":
            if self.supervised and y is not None:
                self.reducer_ = umap.UMAP(
                    n_components=n_used,
                    n_neighbors=self.n_neighbors,
                    min_dist=self.min_dist,
                    target_metric="categorical"
                ).fit(arr, y)
            else:
                self.reducer_ = umap.UMAP(
                    n_components=n_used,
                    n_neighbors=self.n_neighbors,
                    min_dist=self.min_dist
                ).fit(arr)
        elif self._method_norm == "tsne":
            self.reducer_ = TSNE(n_components=n_used)
        elif self._method_norm == "lda":
            self.reducer_ = LinearDiscriminantAnalysis(
                n_components=n_used
            ).fit(arr, y)
        elif self._method_norm == "mds":
            self.reducer_ = MDS(n_components=n_used).fit(arr)
        elif self._method_norm == "isomap":
            self.reducer_ = Isomap(n_components=n_used).fit(arr)

        self.feature_names_out_ = [
            f"{self._method_norm.upper()}{i+1}" for i in range(n_used)
        ]
        return self

    def transform(self, X):
        if self._method_norm is None:
            arr = X.values if isinstance(X, pd.DataFrame) else np.asarray(X, float)
            return arr
        if self.reducer_ is None:
            raise RuntimeError("Must fit before transform")
        arr = X.values if isinstance(X, pd.DataFrame) else np.asarray(X, float)
        if self._method_norm == "tsne":
            return self.reducer_.fit_transform(arr)
        return self.reducer_.transform(arr)

    def get_feature_names_out(self, input_features=None) -> list[str]:
        return self.feature_names_out_
