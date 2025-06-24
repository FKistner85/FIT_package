# src/FIT_python/dim_reduction_utils.py
"""Utility functions for dimensionality reduction."""

from typing import Optional
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

try:
    import umap.umap_ as umap
except Exception:  # pragma: no cover - optional dependency
    umap = None  # type: ignore


def reduce_pca(X: pd.DataFrame, n_components: int = 2) -> pd.DataFrame:
    """Return PCA projection of ``X`` with ``n_components`` dimensions."""
    pca = PCA(n_components=n_components, random_state=0)
    comps = pca.fit_transform(X)
    cols = [f"PC{i+1}" for i in range(n_components)]
    return pd.DataFrame(comps, columns=cols, index=X.index)


def reduce_tsne(X: pd.DataFrame, n_components: int = 2) -> pd.DataFrame:
    """Return t-SNE embedding of ``X`` with ``n_components`` dimensions."""
    tsne = TSNE(n_components=n_components, random_state=0)
    emb = tsne.fit_transform(X)
    cols = [f"TSNE{i+1}" for i in range(n_components)]
    return pd.DataFrame(emb, columns=cols, index=X.index)


def reduce_umap(X: pd.DataFrame, n_components: int = 2) -> pd.DataFrame:
    """Return UMAP embedding of ``X`` with ``n_components`` dimensions."""
    if umap is None:
        raise ImportError("umap-learn is not installed")
    reducer = umap.UMAP(n_components=n_components, random_state=0)
    emb = reducer.fit_transform(X)
    cols = [f"UMAP{i+1}" for i in range(n_components)]
    return pd.DataFrame(emb, columns=cols, index=X.index)
