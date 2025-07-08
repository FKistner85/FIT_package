# feature_selection_wrapper.py

## Overview
Feature selection methods: forward, random forest, variance, univariate and LASSO.

## Key Components
- FeatureSelectionTransformer

### FeatureSelectionTransformer
Offers several strategies to rank and select features including forward search,
random‑forest importance, variance filtering, univariate tests and LASSO. Call
`fit` with the full dataset and target labels to compute a ranking, then
`transform` to reduce the feature matrix. Forward selection can be slow on high
dimensions while variance filtering is much faster but ignores the target. The
unified class enables quick experimentation but results depend heavily on the
underlying method chosen.

## References
- https://scikit-learn.org/stable/modules/feature_selection.html

## Assumptions and Limitations
Ranking length controlled by k; None selects all features.
