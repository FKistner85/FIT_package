# pipeline_wrapper_sex.py

## Overview
Orchestrates import, splitting, summary and model training for sex classification.

## Key Components
- PipelineWrapper
- get_pipeline_steps

### PipelineWrapper
High‑level class that bundles data preparation and model training for sex
classification. Call `prepare()` once to import raw data, create splits and
summaries, then `train()` to evaluate all configured models. The class relies on
paths defined in the configuration module and writes fitted pipelines to disk.
Running many model variants may require significant compute time and storage.

### get_pipeline_steps
Helper that assembles the preprocessing steps according to the selected
hyperparameters. It validates the supplied options and returns a list suitable
for constructing an `sklearn.pipeline.Pipeline`. Mis‑configured parameters raise
`ValueError` so check available choices before calling.

## References
- https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html

## Assumptions and Limitations
Caches pipelines on disk; saves best model per species.
