# geometric_pairwise_projection.py

## Overview
Runs pairwise projections with optional sex model probabilities and caches results.

## Key Components
- generate_pairwise_comparisons_from_df
- run_all_pairwise_projections_parallel

## References
- https://joblib.readthedocs.io/
- https://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.LinearDiscriminantAnalysis.html

## Assumptions and Limitations
Processes comparisons in batches; uses dimensionality reduction before distance computation.
