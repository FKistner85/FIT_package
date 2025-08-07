# geometric_pairwise_projection.py

## Overview
Runs pairwise projections with optional sex model probabilities.

## Key Components
- generate_pairwise_comparisons_from_df
- run_all_pairwise_projections_parallel

### generate_pairwise_comparisons_from_df
Builds a list of trail pairings with fold assignments for cross‑validation. The
result can be stored with joblib to reuse in later experiments. The input
DataFrame must contain individual identifiers and sufficient samples per trail
size.
Set ``evaluation=True`` to skip fold handling when the DataFrame lacks a ``fold``
column.

### run_all_pairwise_projections_parallel
Executes feature selection, dimensionality reduction and distance computation
for each comparison, optionally enriching the data with probabilities from a sex
classification model. Processing is parallelised using joblib and can be
batched via the `batch_size` parameter. The procedure can be computationally
heavy but enables systematic evaluation of many parameter combinations.

## References
- https://joblib.readthedocs.io/
- https://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.LinearDiscriminantAnalysis.html

## Assumptions and Limitations
Processes comparisons in batches; uses dimensionality reduction before distance computation.
