# feature_selection_wrapper.py

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
【F:src/FIT_python/pipeline_individual_id/feature_selection_wrapper.py†L61-L69】

* **Forward selection** greedily adds features that maximise an F-statistic conditioned on previous selections.
* **Random forests** rank features by importance but can favour correlated variables.
* **Variance thresholding** removes features with little spread; simple but may discard informative low-variance ones.
* **Univariate tests** (ANOVA F-score) consider each feature separately and ignore interactions.
* **LASSO** uses $ℓ_1$ regularisation to shrink coefficients; it handles high dimensionality but may underperform with correlated predictors.
