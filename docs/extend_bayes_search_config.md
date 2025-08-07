# Extending the BayesSearchCV Configuration

This short guide explains how to add more preprocessing options and classifiers to the sex
classification search. All settings live in `CONFIG` and are consumed by
`search.py` when constructing `SEARCH_SPACES`.

## 1. Update `CONFIG`

Add new entries under `pipeline_sex.search_spaces` so the configuration lists
every parameter you would like to explore. Example:

```python
CONFIG["pipeline_sex"]["search_spaces"].update(
    {
        "outlier": [None, "clip", "zscore"],
        "scale": [None, "standard", "robust"],
        "reduce_pre__method": [None, "pca", "umap", "lda"],
        "reduce_post__method": [None, "pca", "umap", "lda"],
    }
)
```

The lists can be expanded further if additional transformers become available.

## 2. Extend `SEARCH_SPACES`

`search.py` reads these lists and converts them to `skopt.space.Categorical`
objects. Create one entry per parameter and supply multiple classifiers via
`EstimatorWrapper`:

```python
SEARCH_SPACES = {
    "outlier": Categorical(
        [
            OutlierCleanerTransformer(method="clip"),
            OutlierCleanerTransformer(method="zscore"),
        ],
        transform="identity",
    ),
    "scale": Categorical(
        [
            FeatureScalerTransformer(method="standard"),
            FeatureScalerTransformer(method="robust"),
        ],
        transform="identity",
    ),
    "reduce_pre__method": Categorical(SEARCH_SPACE_CFG["reduce_pre__method"]),
    "reduce_post__method": Categorical(SEARCH_SPACE_CFG["reduce_post__method"]),
    "select__method": Categorical(SEARCH_SPACE_CFG["select__method"]),
    "select__k": Categorical(SEARCH_SPACE_CFG["select__k"]),
    "clf": Categorical(
        [EstimatorWrapper(MODELS[k]) for k in ("rf_small", "rf_med", "xgb_std", "lda")],
        transform="identity",
    ),
}
```

The functions `run_otter_search_sex` and `run_species_search` also look up
``reuse_results`` in ``CONFIG['pipeline_sex']['run_otter_search_sex']``. Set
this to ``True`` to load existing CSV files instead of running a new search.

`BayesSearchCV` will now explore combinations of outlier handling, scaling,
dimensionality reduction and several classifiers in a single optimisation run.
`get_pipeline_steps` already accepts the corresponding arguments, so no further
changes are necessary.
