"""Preconfigured dimensionality reducers for pipelines."""

from .dimensionality_reduction_wrapper import DimensionalityReducerTransformer

REDUCERS = {
    "pca_10": DimensionalityReducerTransformer(method="pca", n_components=10),
    "umap_10": DimensionalityReducerTransformer(
        method="umap", n_components=10, n_neighbors=15, min_dist=0.1
    ),
    "tsne_2": DimensionalityReducerTransformer(method="tsne", n_components=2),
}
