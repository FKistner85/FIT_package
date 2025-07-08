# split_utils.py

## Overview
Utility functions for stratified and group-based train/test splits.

## Key Components
- stratified_individual_split
- train_test_group_split
- group_stratified_kfold
- ensure_valid_splits

## References
- https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedGroupKFold.html

## Assumptions and Limitations
Requires group and stratify columns; ensures balanced folds.
