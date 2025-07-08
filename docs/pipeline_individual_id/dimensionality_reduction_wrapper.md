# dimensionality_reduction_wrapper.py

The `DimensionalityReducerTransformer` bundles several algorithms for reducing feature space dimensionality. Available methods are listed in the class docstring:

```python
    Wrapper für Dimensionsreduktion:
      - PCA (unsupervised)
      - UMAP (unsupervised + supervised)
      - t-SNE (unsupervised)
      - LDA (supervised)
      - MDS (unsupervised)
      - Isomap (unsupervised)
```
【F:src/FIT_python/pipeline_individual_id/dimensionality_reduction_wrapper.py†L13-L21】

* **PCA** reduces variance along orthogonal axes and is simple to interpret, but only captures linear structure.
* **UMAP** preserves local topology using manifolds and can run in supervised mode; it handles non-linear patterns well but may distort global distances.
* **t-SNE** excels at visualising clusters in two or three dimensions, yet the embedding is stochastic and not easily invertible.
* **LDA** finds axes that best separate known classes; it requires labelled data and assumes Gaussian class distributions.
* **MDS** and **Isomap** both attempt to preserve pairwise distances; they can be computationally expensive for large datasets.

See [McInnes et al., 2018](https://arxiv.org/abs/1802.03426) for UMAP and [van der Maaten & Hinton, 2008](https://jmlr.org/papers/v9/vandermaaten08a.html) for t-SNE.
