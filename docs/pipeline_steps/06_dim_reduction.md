# Step 6: Dimensionality Reduction

`create_dim_reduction.py` performs PCA, but the wrapper
`reduce_all` in `dim_reduction_wrapper` also supports t-SNE and UMAP if
those libraries are installed.

**Selectable Methods**
- `PCA` – linear projection via singular value decomposition.
- `TSNE` – t-distributed stochastic neighbour embedding [[Maaten08]].
- `UMAP` – uniform manifold approximation and projection [[McInnes18]].

**Algorithmic Cost**
- PCA: O(min(n p^2, p n^2))
- t-SNE: roughly O(n^2)
- UMAP: approximately O(n log n)

[Maaten08]: https://doi.org/10.1007/978-3-540-74958-5_7
[McInnes18]: https://arxiv.org/abs/1802.03426

