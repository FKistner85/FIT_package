# id_config.py

`PairwiseEstimator` wraps `run_all_pairwise_projections_parallel` so it can be used inside `BayesSearchCV`:

```python
class PairwiseEstimator:
    def __init__(
        self,
        feature_cols: Iterable[str],
        *,
        outlier_method=None,
        scaler_method=None,
        selection_method=None,
        k_features=5,
        reducer="pca",
        n_components=2,
        use_sexmodel_prediction=False,
        sexmodel_path=None,
    ):
        ...

    def predict(self, X: pd.DataFrame):
        comps, _ = generate_pairwise_comparisons_from_df(X)
        res = run_all_pairwise_projections_parallel(
            ...,
            use_sexmodel_prediction=self.use_sexmodel_prediction,
            sexmodel_path=self.sexmodel_path,
        )
        return pd.DataFrame(res)
```
【F:src/FIT_python/pipeline_individual_id/id_config.py†L48-L121】

The Bayesian search space includes ``"est__use_sexmodel_prediction"`` to toggle
appending sex-model predictions during optimisation.

The helper `run_species_search()` loads the training splits and performs a Bayesian hyperparameter search. The cross-validation strategy can be customised via the ``cv`` parameter. When set to ``"fold"`` (default) the function constructs a :class:`~sklearn.model_selection.PredefinedSplit` from the ``Fold`` column. Results are written to ``results/<species>_id_search/cv_results.csv``.

Example usage:

```python
from FIT_python.pipeline_individual_id import run_id_search
run_id_search(n_iter=10, random_state=0, cv="fold")
```

After optimisation one can inspect the Euclidean distances via
`search.best_estimator_.named_steps['est'].results_`.
