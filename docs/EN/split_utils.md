# split_utils.py

## Overview
Utility functions for stratified and group-based train/test splits.

## Key Components
- stratified_individual_split
- train_test_group_split
- group_stratified_kfold
- ensure_valid_splits

### stratified_individual_split
Splits a DataFrame by individual while keeping the class distribution of the
`stratify_col` (typically `sex`) balanced between train and test. Rows without a
valid group or label are placed in an inference set. If ``add_folds=True`` the
returned training set also contains a ``Fold`` column generated via
``StratifiedGroupKFold`` so that each individual occurs in a single fold and the
sex ratio is preserved. This requires enough individuals per class.

### train_test_group_split
Randomly assigns entire groups to either the training or test set without
stratification. Useful for quick experiments where class balance is less
critical. If groups vary greatly in size the resulting split may be skewed.

### group_stratified_kfold
Applies `StratifiedGroupKFold` to create a fold assignment that preserves class
ratios within each fold while keeping groups intact. The function retries with
different seeds when the class distribution cannot be satisfied. It requires at
least two groups per class.

### ensure_valid_splits
Checks all generated splits on disk for a proper `Fold` column that contains
both classes in every fold. Any problems trigger warnings so that the user can
recreate the splits if necessary.

## References
- https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedGroupKFold.html

## Assumptions and Limitations
Requires group and stratify columns; ensures balanced folds.
