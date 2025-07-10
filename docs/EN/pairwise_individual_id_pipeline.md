# pairwise_individual_id_pipeline.py

## Overview
Main pipeline computing pairwise distances after feature selection and reduction.

## Key Components
- run_all_pairwise_projections_parallel
- run_embedding_once_pipeline

### run_all_pairwise_projections_parallel
Handles end‑to‑end processing of all trail comparisons. The function performs
outlier removal, scaling, feature selection and dimensionality reduction before
computing multiple distance metrics. Pass a list of comparisons generated from
`generate_trails_and_trailpairs` along with the base DataFrame. Extensive
parameter combinations can lead to long runtimes so parallel execution via
`n_jobs` is recommended.

### run_embedding_once_pipeline
Embeds all training trails a single time and only transforms each test trail
before measuring distances. Useful when the same reference set is reused
across many comparisons.

## References
- https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html

## Assumptions and Limitations
Optionally enriches data with sex-model probabilities.
