# src/FIT_python/pipeline/dimensionality_reduction_wrapper.py

import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, MDS, Isomap
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import umap


class DimensionalityReducerTransformer(TransformerMixin, BaseEstimator):
    """Apply various dimensionality reduction techniques.

    Supported methods
    -----------------
    - PCA (unsupervised)
    - UMAP (unsupervised or supervised)
    - t-SNE (unsupervised)
    - LDA (supervised)
    - MDS (unsupervised)
    - Isomap (unsupervised)
    """

    def __init__(
        self,
        method: str = "pca",
        n_components: int = 2,
        supervised: bool = False,
        **kwargs,
    ):
        """Store parameters verbatim for sklearn cloning."""

        self.method = method
        self.n_components = n_components
        self.supervised = supervised
        self.kwargs = kwargs

        method_norm = method.lower() if method is not None else None
        if method_norm not in (None, "pca", "umap", "tsne", "lda", "mds", "isomap"):
            raise ValueError(f"Unknown method: {method!r}")
        self._method_norm = method_norm
        self.requested_n = n_components
        self.reducer_ = None
        self.feature_names_out_: list[str] = []

    def fit(self, X, y=None):
        arr = X.values if isinstance(X, pd.DataFrame) else np.asarray(X, dtype=float)
        n_samples, n_features = arr.shape
        max_c = min(n_samples, n_features)
        n_used = min(self.requested_n, max_c)

        if self._method_norm in ("umap", "lda") and self.supervised and y is None:
            raise ValueError(
                f"Supervised {self._method_norm.upper()} requires target labels `y`."
            )

        if self._method_norm == "pca":
            reducer = PCA(n_components=n_used, **self.kwargs)
            reducer.fit(arr)

        elif self._method_norm == "umap":
            if self.supervised and y is not None:
                reducer = umap.UMAP(
                    n_components=n_used, target_metric="categorical", **self.kwargs
                )
                reducer.fit(arr, y)
            else:
                reducer = umap.UMAP(n_components=n_used, **self.kwargs)
                reducer.fit(arr)

        elif self._method_norm == "lda":
            reducer = LinearDiscriminantAnalysis(n_components=n_used, **self.kwargs)
            reducer.fit(arr, y)

        elif self._method_norm == "tsne":
            algo = "barnes_hut" if n_used <= 3 else "exact"
            reducer = TSNE(n_components=n_used, method=algo, **self.kwargs)

        elif self._method_norm == "mds":
            reducer = MDS(n_components=n_used, **self.kwargs)
            reducer.fit(arr)

        elif self._method_norm == "isomap":
            reducer = Isomap(n_components=n_used, **self.kwargs)
            reducer.fit(arr)

        self.reducer_ = reducer
        self.feature_names_out_ = [
            f"{self._method_norm.upper()}{i+1}" for i in range(n_used)
        ]
        return self

    def transform(self, X):
        if self.reducer_ is None:
            raise RuntimeError(
                "DimensionalityReducerTransformer must be fitted before transform."
            )
        arr = X.values if isinstance(X, pd.DataFrame) else np.asarray(X, dtype=float)

        if self._method_norm in ("pca", "umap", "lda", "mds", "isomap"):
            return self.reducer_.transform(arr)

        elif self._method_norm == "tsne":
            return self.reducer_.fit_transform(arr)

    def get_feature_names_out(self, input_features=None) -> list[str]:
        return self.feature_names_out_


# Standard presets for typical reducer settings
REDUCER_PRESETS = {
    "pca_10": DimensionalityReducerTransformer(method="pca", n_components=10),
    "umap_10": DimensionalityReducerTransformer(method="umap", n_components=10),
    "tsne_2": DimensionalityReducerTransformer(method="tsne", n_components=2),
}
