# outlier_wrapper.py

## Overview
Cleans outliers using quantile clipping or z-score bounds.

## Key Components
- OutlierCleanerTransformer

### OutlierCleanerTransformer
Removes extreme values using either percentile clipping or z‑score limits.
Fitting computes the necessary bounds from the training data. Use this
transformer early in a preprocessing pipeline to reduce the influence of
outliers. Clipping is simple but may discard valid extreme observations whereas
the z‑score method assumes an approximately normal distribution.

## References
- https://scikit-learn.org/stable/modules/preprocessing.html#robust-scaler

## Assumptions and Limitations
Assumes numeric inputs; returns same type as input.
