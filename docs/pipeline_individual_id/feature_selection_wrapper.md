# feature_selection_wrapper.py

This module was moved to the shared ``pipeline_general`` package. The
documentation remains here for backwards compatibility.

`FeatureSelectionTransformer` supports several strategies. Initialisation restricts the method to one of the following:

```python
    def __init__(
        self,
        method: str = None,  # 'forward', 'random_forest', 'variance', 'univariate', 'lasso', or None (use all)
        k: int = None,
        random_state: int = 0
    ):
        allowed_methods = [None, 'forward', 'random_forest', 'variance', 'univariate', 'lasso']
        if method not in allowed_methods:
            raise ValueError(f"method must be one of {allowed_methods}")
```
【F:src/FIT_python/pipeline_general/feature_selection_wrapper.py†L60-L71】

* **Forward selection** greedily adds features that maximise an F‑statistic conditioned on previously chosen variables. It is conceptually simple and dates back to classical regression modelling (*Draper & Smith, 1966*), but the sequential nature can lead to sub‑optimal global solutions.

* **Random forests** rank features by importance based on ensembles of decision trees (*Breiman, 2001*). They work well with nonlinear relationships and handle mixed feature types. However, they may favour highly correlated variables and can be unstable for very small sample sizes.

* **Variance thresholding** removes features with little spread. This filter is computationally cheap and unsupervised, yet it might discard low‑variance features that are actually predictive when combined with others.

* **Univariate tests** such as the ANOVA F‑score examine each feature independently (*Fisher, 1925*). They are fast and easy to interpret, but they ignore interactions and multicollinearity among variables.

* **LASSO** applies $\ell_1$ regularisation to regression coefficients (*Tibshirani, 1996*). It is well suited to high‑dimensional settings and produces sparse solutions, though it can struggle when important predictors are strongly correlated.

`FeatureSelectionTransformer` returns the selected subset as a DataFrame or array and stores a ranking of all examined features for later inspection.

**References**
* Breiman, L. (2001). "Random Forests." *Machine Learning*.
* Draper, N., & Smith, H. (1966). *Applied Regression Analysis*. Wiley.
* Fisher, R. A. (1925). *Statistical Methods for Research Workers*.
* Tibshirani, R. (1996). "Regression shrinkage and selection via the lasso." *Journal of the Royal Statistical Society, Series B*.
