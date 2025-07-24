# id_config.py

`PairwiseEstimator` wraps `run_all_pairwise_projections_parallel` so it can be used inside `BayesSearchCV`:

```python
class PairwiseEstimator:
    def __init__(self, feature_cols: Iterable[str], *, outlier_method=None,
                 scaler_method=None, selection_method=None, k_features=5,
                 reducer="pca", n_components=2):
        ...
    def predict(self, X: pd.DataFrame):
        comps, _ = generate_pairwise_comparisons_from_df(X)
        res = run_all_pairwise_projections_parallel(...)
        return pd.DataFrame(res)
```
【F:src/FIT_python/pipeline_individual_id/id_config.py†L32-L71】

The helper `run_species_search()` loads the training splits, constructs a `PredefinedSplit` from the fold numbers and performs a Bayesian hyperparameter search. Results are written to `results/<species>_id_search/cv_results.csv`.

Example usage:

```python
from FIT_python.pipeline_individual_id import run_id_search
run_id_search(n_iter=10, random_state=0, reuse_results=True)
```

When ``reuse_results`` is ``True`` and result files already exist, the search
is skipped and the CSV tables are loaded instead.

After optimisation one can inspect the Euclidean distances via
`search.best_estimator_.named_steps['est'].results_`.
