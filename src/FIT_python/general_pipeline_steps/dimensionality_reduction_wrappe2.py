# src/FIT_python/pipeline_sex/dimensionality_reduction_wrapper.py

import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, MDS, Isomap
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import umap
from FIT_python.soft_config import SOFT_CONFIG


class DimensionalityReducerTransformer(TransformerMixin, BaseEstimator):
    """Apply different dimensionality reduction techniques.

    Supported methods
    -----------------
    - ``None``: identity transformation
    - PCA
    - UMAP
    - t-SNE
    - LDA
    - MDS
    - Isomap
    """

    def __init__(
        self,
        method: str | None = SOFT_CONFIG["general_pipeline_steps"]["dim_reducer_defaults"]["method"],
        n_components: int = SOFT_CONFIG["general_pipeline_steps"]["dim_reducer_defaults"]["n_components"],
        supervised: bool = False,
        n_neighbors: int = SOFT_CONFIG["general_pipeline_steps"]["dim_reducer_defaults"]["n_neighbors"],
        min_dist: float = SOFT_CONFIG["general_pipeline_steps"]["dim_reducer_defaults"]["min_dist"],
        whiten: bool = SOFT_CONFIG["general_pipeline_steps"]["dim_reducer_defaults"]["whiten"],
        **kwargs,
    ):
        # raw parameters for cloning
        self.method = method
        self.n_components = n_components
        self.supervised = supervised
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        self.whiten = whiten
        self.kwargs = kwargs

        # intern normalisiert
        self._method_norm = method.lower() if method is not None else None
        if self._method_norm not in (
            None,
            "pca",
            "umap",
            "tsne",
            "lda",
            "mds",
            "isomap",
        ):
            raise ValueError(f"Unknown method: {method!r}")

        self.requested_n = n_components
        self.reducer_ = None
        self.feature_names_out_: list[str] = []

    def get_params(self, deep=True):
        # return exactly the init parameters expected by RandomizedSearchCV
        return {
            "method": self.method,
            "n_components": self.n_components,
            "supervised": self.supervised,
            "n_neighbors": self.n_neighbors,
            "min_dist": self.min_dist,
            "whiten": self.whiten,
            **self.kwargs,
        }

    def fit(self, X, y=None):
        # Identity-Fall
        if self._method_norm is None:
            cols = X.columns.tolist() if isinstance(X, pd.DataFrame) else None
            self.feature_names_out_ = cols or [f"f{i}" for i in range(X.shape[1])]
            self.reducer_ = None
            return self

        arr = X.values if isinstance(X, pd.DataFrame) else np.asarray(X, dtype=float)
        n_samples, n_features = arr.shape
        n_used = min(self.requested_n, n_samples, n_features)

        if self._method_norm == "pca":
            reducer = PCA(n_components=n_used, whiten=self.whiten, **self.kwargs).fit(
                arr
            )

        elif self._method_norm == "umap":
            umap_params = {
                "n_components": n_used,
                "n_neighbors": self.n_neighbors,
                "min_dist": self.min_dist,
                **self.kwargs,
            }
            if self.supervised and y is not None:
                reducer = umap.UMAP(target_metric="categorical", **umap_params).fit(
                    arr, y
                )
            else:
                reducer = umap.UMAP(**umap_params).fit(arr)

        elif self._method_norm == "tsne":
            reducer = TSNE(n_components=n_used, **self.kwargs)

        elif self._method_norm == "lda":
            reducer = LinearDiscriminantAnalysis(
                n_components=n_used, **self.kwargs
            ).fit(arr, y)

        elif self._method_norm == "mds":
            reducer = MDS(n_components=n_used, **self.kwargs).fit(arr)

        elif self._method_norm == "isomap":
            reducer = Isomap(n_components=n_used, **self.kwargs).fit(arr)

        else:
            raise ValueError(f"Unhandled reduction method: {self.method!r}")

        self.reducer_ = reducer
        self.feature_names_out_ = [
            f"{self._method_norm.upper()}{i+1}" for i in range(n_used)
        ]
        return self

    def transform(self, X):
        if self._method_norm is None:
            return (
                X.values if isinstance(X, pd.DataFrame) else np.asarray(X, dtype=float)
            )

        if self.reducer_ is None:
            raise RuntimeError(
                "DimensionalityReducerTransformer must be fitted before transform."
            )

        arr = X.values if isinstance(X, pd.DataFrame) else np.asarray(X, dtype=float)
        if self._method_norm in ("pca", "umap", "lda", "mds", "isomap"):
            return self.reducer_.transform(arr)
        else:  # tsne
            return self.reducer_.fit_transform(arr)

    def get_feature_names_out(self, input_features=None) -> list[str]:
        return self.feature_names_out_


# Standard presets for typical reducer settings
REDUCER_PRESETS = {
    "pca_10": DimensionalityReducerTransformer(method="pca", n_components=10),
    "umap_10": DimensionalityReducerTransformer(
        method="umap", n_components=10, n_neighbors=15, min_dist=0.1
    ),
    "tsne_2": DimensionalityReducerTransformer(method="tsne", n_components=2),
}
