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

* **PCA** (principal component analysis, *Pearson 1901; Hotelling 1933*) rotates the feature space to directions of maximal variance. It is straightforward to compute and interpret, but only captures linear structure and may require scaling of the inputs.

* **UMAP** (*McInnes et al., 2018*) preserves local neighbourhoods via a manifold assumption. It can operate in unsupervised or supervised mode and works well for non‑linear embeddings. Global distances, however, can be distorted and the result depends on several hyperparameters.

* **t‑SNE** (*van der Maaten & Hinton, 2008*) excels at visualising clusters in two or three dimensions by converting distances into conditional probabilities. The embedding is stochastic and not naturally extendable to new samples, which limits its use beyond exploratory analysis.

* **LDA** (*Fisher, 1936*) finds axes that best separate pre‑defined classes. It is a supervised method that assumes Gaussian class distributions and equal covariances. When these assumptions are violated, performance may degrade.

* **MDS** and **Isomap** aim to preserve pairwise distances or geodesic distances respectively. Classical MDS (*Kruskal, 1964*) can reveal low‑dimensional structure but scales poorly with many samples. Isomap (*Tenenbaum et al., 2000*) extends MDS by approximating geodesics via a neighbourhood graph and is useful for unfolding non‑linear manifolds.

The transformer exposes a unified interface so that different methods can be swapped by a parameter, returning transformed arrays suitable for further modelling or visualisation.

**References**
* Fisher, R. A. (1936). "The use of multiple measurements in taxonomic problems." *Annals of Eugenics*.
* Hotelling, H. (1933). "Analysis of a complex of statistical variables into principal components." *Journal of Educational Psychology*.
* Kruskal, J. B. (1964). "Multidimensional scaling by optimizing goodness of fit to a nonmetric hypothesis." *Psychometrika*.
* McInnes, L., Healy, J., & Melville, J. (2018). "UMAP: Uniform Manifold Approximation and Projection for dimension reduction." arXiv:1802.03426.
* Pearson, K. (1901). "On lines and planes of closest fit to systems of points in space." *Philosophical Magazine*.
* Tenenbaum, J. B., de Silva, V., & Langford, J. C. (2000). "A global geometric framework for nonlinear dimensionality reduction." *Science*.
* van der Maaten, L., & Hinton, G. (2008). "Visualizing data using t‑SNE." *Journal of Machine Learning Research*.

See [McInnes et al., 2018](https://arxiv.org/abs/1802.03426) for UMAP and [van der Maaten & Hinton, 2008](https://jmlr.org/papers/v9/vandermaaten08a.html) for t-SNE.
