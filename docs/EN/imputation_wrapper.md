# imputation_wrapper.py

## Overview
Wrapper around IterativeImputer with RandomForestRegressor.

## Key Components
- ImputationWrapper

### ImputationWrapper
Applies scikit‑learn's `IterativeImputer` with a `RandomForestRegressor` base
estimator to fill missing numerical values. Call `fit` on the training data to
learn imputation patterns and then `transform` on any data set needing the same
treatment. The approach captures non‑linear relations but can be computationally
heavy and may yield slightly different results across runs due to randomness.

## References
- https://scikit-learn.org/stable/modules/generated/sklearn.impute.IterativeImputer.html

## Assumptions and Limitations
Imputes numeric columns; fitted on numeric subset of DataFrame.
