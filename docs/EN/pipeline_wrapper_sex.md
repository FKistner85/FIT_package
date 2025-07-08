# pipeline_wrapper_sex.py

## Overview
Orchestrates import, splitting, summary and model training for sex classification.

## Key Components
- PipelineWrapper
- get_pipeline_steps

## References
- https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html

## Assumptions and Limitations
Caches pipelines on disk; saves best model per species.
