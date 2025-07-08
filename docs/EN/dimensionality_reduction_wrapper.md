# dimensionality_reduction_wrapper.py

## Overview
Transformer implementing PCA, UMAP, t-SNE, LDA, MDS and Isomap.

## Key Components
- DimensionalityReducerTransformer

### DimensionalityReducerTransformer
Sklearn‑style transformer that wraps several reduction algorithms such as PCA,
UMAP, t‑SNE, LDA, MDS and Isomap. Select the desired method via the `method`
parameter and optionally enable the `supervised` flag when a label vector is
required (e.g. for LDA or supervised UMAP). The transformer expects numeric
matrices and returns the reduced coordinates. While PCA and LDA are relatively
fast, methods like t‑SNE can be slow and non‑deterministic. The unified API
makes experimentation easy but not all approaches work equally well for all
datasets.

## References
- https://scikit-learn.org/stable/modules/decomposition.html
- https://umap-learn.readthedocs.io/en/latest/

## Assumptions and Limitations
Chooses method based on parameters; supervised mode for LDA/UMAP.
