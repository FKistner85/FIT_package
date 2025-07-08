# feature_scaler_wrapper.py

## Overview
Scaler transformer wrapping StandardScaler or RobustScaler.

## Key Components
- FeatureScalerTransformer

### FeatureScalerTransformer
Scales numeric features using either the standard or robust method from
scikit‑learn. The transformer keeps track of the column order so that a
`DataFrame` passed to `transform` returns another `DataFrame` with the same
labels. Use this component when models are sensitive to feature scales. Be aware
that scaling parameters are learned from the training set and may not be
appropriate for out‑of‑distribution data.

## References
- https://scikit-learn.org/stable/modules/preprocessing.html#scaling-features

## Assumptions and Limitations
Returns array or DataFrame matching the input type.
