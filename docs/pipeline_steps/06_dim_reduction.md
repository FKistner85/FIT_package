# Step 6: Dimensionality Reduction

`create_dim_reduction.py` performs principal component analysis (PCA)
on the selected features.  The wrapper `reduce_all` in
`dim_reduction_wrapper` also exposes t-SNE and UMAP if the respective
libraries are available.

**Selectable Methods**
- `PCA` – a linear method that finds orthogonal axes capturing maximum
  variance.  Implemented via singular value decomposition.  Useful for
  noise reduction and exploratory analysis.
- `TSNE` – t-distributed stochastic neighbour embedding [[Maaten08]]; a
  non-linear technique that converts pairwise distances into
  probabilities and optimises them using gradient descent.  Often used
  for visualisation of clusters.
- `UMAP` – uniform manifold approximation and projection [[McInnes18]],
  another non-linear method which builds a fuzzy topological
  representation of the data and optimises a low-dimensional layout.

| Method | Idea | Complexity |
| ------ | ---- | ---------- |
| PCA | Linear projection via SVD | O(min(n × p², p × n²)) |
| t-SNE | Probabilistic embedding with gradient descent | ~O(n²) |
| UMAP | Fuzzy simplicial set projection | ~O(n log n) |