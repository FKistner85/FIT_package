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


def reduce_pca(X: pd.DataFrame, n_components: int = 2, *, random_state: int = 0) -> pd.DataFrame:
    """Return PCA projection of ``X`` with ``n_components`` dimensions."""
    pca = PCA(n_components=n_components, random_state=random_state)
    comps = pca.fit_transform(X)
    cols = [f"PC{i+1}" for i in range(n_components)]
    return pd.DataFrame(comps, columns=cols, index=X.index)


def reduce_tsne(
    X: pd.DataFrame,
    n_components: int = 2,
    *,
    perplexity: float = 30.0,
    random_state: int = 0,
) -> pd.DataFrame:
    """Return t-SNE embedding of ``X`` with ``n_components`` dimensions."""
    tsne = TSNE(n_components=n_components, perplexity=perplexity, random_state=random_state)
    emb = tsne.fit_transform(X)
    cols = [f"TSNE{i+1}" for i in range(n_components)]
    return pd.DataFrame(emb, columns=cols, index=X.index)


def reduce_umap(
    X: pd.DataFrame,
    n_components: int = 2,
    *,
    n_neighbors: int = 15,
    random_state: int = 0,
) -> pd.DataFrame:
    """Return UMAP embedding of ``X`` with ``n_components`` dimensions."""
    if umap is None:
        raise ImportError("umap-learn is not installed")
    reducer = umap.UMAP(n_components=n_components, n_neighbors=n_neighbors, random_state=random_state)
    emb = reducer.fit_transform(X)
    cols = [f"UMAP{i+1}" for i in range(n_components)]
    return pd.DataFrame(emb, columns=cols, index=X.index)
