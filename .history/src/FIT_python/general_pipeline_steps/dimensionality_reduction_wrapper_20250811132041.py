# src/FIT_python/pipeline/dimensionality_reduction_wrapper.py

import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, MDS, Isomap
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import umap
from FIT_python.utils import debug_report
from FIT_python.config import CONFIG, DEFAULT_METADATA_COLS

# Metadata columns that should be passed through unchanged when the input is a
# DataFrame. These columns are ignored during dimensionality reduction. The
# actual list is read from :data:`CONFIG` so pipelines can override it.


class DimensionalityReducerTransformer(TransformerMixin, BaseEstimator):
    """Apply various dimensionality reduction techniques.

    Supported methods
    -----------------
    - PCA (unsupervised)
    - UMAP (supervised)
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
        """Initialise the reducer.

        Parameters
        ----------
        method:
            Reduction algorithm. Options: ``'pca'``, ``'umap'``, ``'tsne'``,
            ``'lda'``, ``'mds'`` or ``'isomap'``. Defaults to ``'pca'``.
        n_components:
            Number of dimensions to keep. Defaults to ``2``.
        supervised:
            Whether to perform supervised reduction where supported. This flag
            is ignored when ``method='umap'`` because UMAP is always trained in
            supervised mode.
        **kwargs:
            Additional arguments passed to the underlying reducer.
        """

        self.method = method
        self.n_components = n_components
        self.supervised = supervised
        self.kwargs = kwargs

        method_norm = method.lower() if method is not None else None
        if method_norm not in (None, "pca", "umap", "tsne", "lda", "mds", "isomap"):
            raise ValueError(f"Unknown method: {method!r}")
        self._method_norm = method_norm
        if method_norm == "umap":
            # Always run UMAP in supervised mode regardless of the flag
            self.supervised = True
        self.requested_n = n_components
        self.reducer_ = None
        self.input_features_: list[str] = []
        self.feature_names_out_: list[str] = []

    def fit(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            self.metadata_cols_ = [c for c in X.columns
                                if c in DEFAULT_METADATA_COLS or not pd.api.types.is_numeric_dtype(X[c])]
            X_num = X.drop(columns=self.metadata_cols_, errors="ignore")
            arr = X_num.to_numpy(dtype=float)
            self.input_features_ = X_num.columns.to_list()
        else:
            arr = np.asarray(X, dtype=float)
            self.input_features_ = [f"x{i}" for i in range(arr.shape[1])]
            self.metadata_cols_ = []

        # Passthrough, wenn keine Reduktionsmethode gewählt ist
        if self._method_norm is None:
            from sklearn.preprocessing import FunctionTransformer
            transformer = FunctionTransformer(validate=False)
            transformer.fit(arr)
            self.reducer_ = transformer
            self.feature_names_out_ = list(self.input_features_)
            return self

        n_samples, n_features = arr.shape
        max_c = min(n_samples, n_features)

        # requested_n robust behandeln (None -> sinnvoller Default)
        req = getattr(self, "requested_n", None)
        if req is None:
            # Fallback: n_components aus kwargs oder 2 (konservativ)
            req = self.kwargs.get("n_components", 2)
        try:
            req = int(req)
        except (TypeError, ValueError):
            req = 2  # letzter Fallback, falls etwas Ungültiges ankommt

        # Vorläufige Begrenzung durch Datengeometrie
        n_used = max(1, min(req, max_c))

        # Supervised-Reducer benötigen y
        if self._method_norm in ("umap", "lda") and self.supervised and y is None:
            raise ValueError(
                f"Supervised {self._method_norm.upper()} requires target labels `y`."
            )

        if self._method_norm == "pca":
            reducer = PCA(n_components=n_used, **self.kwargs)
            reducer.fit(arr)

        elif self._method_norm == "umap":
            y_series = pd.Series(y)
            if not np.issubdtype(y_series.dtype, np.number):
                y_encoded = pd.factorize(y_series)[0]
            else:
                y_encoded = y_series.to_numpy()
            reducer = umap.UMAP(
                n_components=n_used, target_metric="categorical", **self.kwargs
            )
            reducer.fit(arr, y_encoded)

        elif self._method_norm == "lda":
            # LDA-Constraint: n_components <= min(n_features, n_classes-1)
            classes = np.unique(y)
            n_classes = len(classes)
            # Mindestens 2 Klassen vorausgesetzt; bei n_classes==1 bleibt n_used=1
            n_used = max(1, min(n_used, n_classes - 1))
            reducer = LinearDiscriminantAnalysis(n_components=n_used, **self.kwargs)
            reducer.fit(arr, y)

        elif self._method_norm == "tsne":
            algo = "barnes_hut" if n_used <= 3 else "exact"
            reducer = TSNE(n_components=n_used, method=algo, **self.kwargs)
            # Hinweis: TSNE hat typischerweise nur fit_transform; wir belassen das Verhalten hier unverändert.

        elif self._method_norm == "mds":
            reducer = MDS(n_components=n_used, **self.kwargs)
            reducer.fit(arr)

        elif self._method_norm == "isomap":
            reducer = Isomap(n_components=n_used, **self.kwargs)
            reducer.fit(arr)

        self.reducer_ = reducer
        self.feature_names_out_ = [f"{self._method_norm.upper()}{i+1}" for i in range(n_used)]
        return self


    def transform(self, X):
        if self.reducer_ is None:
            raise RuntimeError(
                "DimensionalityReducerTransformer must be fitted before transform."
            )
        if isinstance(X, pd.DataFrame):
            arr = X[self.input_features_].to_numpy(dtype=float)
        else:
            arr = np.asarray(X, dtype=float)

        if self._method_norm in ("pca", "umap", "lda", "mds", "isomap"):
            out = self.reducer_.transform(arr)
        elif self._method_norm == "tsne":
            out = self.reducer_.fit_transform(arr)
        else:
            out = arr
        debug_report(out, "reduce")
        if out.shape[1] < len(self.feature_names_out_):
            pad = np.zeros((out.shape[0], len(self.feature_names_out_) - out.shape[1]))
            out = np.concatenate([out, pad], axis=1)
        if isinstance(X, pd.DataFrame):
            df_meta = X[[c for c in self.metadata_cols_ if c in X.columns]].copy() if self.metadata_cols_ else pd.DataFrame(index=X.index)
            df_out = pd.DataFrame(out, columns=self.feature_names_out_, index=X.index)
            return pd.concat([df_meta, df_out], axis=1)
        return out

    def get_feature_names_out(self, input_features=None) -> list[str]:
        return self.feature_names_out_


# Standard presets for typical reducer settings
REDUCER_PRESETS = {
    "pca_10": DimensionalityReducerTransformer(method="pca", n_components=10),
    "umap_10": DimensionalityReducerTransformer(method="umap", n_components=10),
    "tsne_2": DimensionalityReducerTransformer(method="tsne", n_components=2),
}