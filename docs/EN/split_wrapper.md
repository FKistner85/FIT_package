# split_wrapper.py

## Overview
High-level interface that loads cleaned data and writes train/test folds.

## Key Components
- SplitWrapper.split_all

### SplitWrapper.split_all
Runs the full splitting workflow: it cleans raw data, applies species‑specific
logic and writes train, test and inference sets for each dataset. Folds are
created using robust heuristics if not already present. Invoke this method once
before training models. Large datasets may produce many files and re‑running the
function will overwrite existing outputs.

## References
- https://scikit-learn.org/stable/modules/cross_validation.html

## Assumptions and Limitations
Creates parquet and CSV outputs per species.
