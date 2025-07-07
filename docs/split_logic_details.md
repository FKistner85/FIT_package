# Train/Test Split Logic

This document describes the data splitting strategy implemented in `src/FIT_python/data_split_and_summary/split_wrapper.py` and the helper functions found in `src/FIT_python/data_split_and_summary/split_utils.py`.

## 1. Species-Specific Splits

`SplitWrapper.split_all()` loads cleaned data sets using `DataImporter` and writes species-specific directories under `data/splits`. Each directory contains three Parquet files:

- `train.parquet`
- `test.parquet`
- `inference.parquet` (optional)

The wrapper ensures that all footprints from the same individual appear **only in one split**. This grouping is handled by the helper `stratified_individual_split` which performs the following steps:

1. Validate that both `individual_id` (group column) and `sex` (stratify column) exist.
2. Generate a meta table with one entry per individual and its sex label.
3. Apply `train_test_split` on the *individuals* while stratifying by sex to preserve the overall sex ratio.
4. Extract the rows belonging to the resulting training and test individual sets. Remaining individuals without valid sex information are assigned to the `inference` set.

As a result, each individual’s footprints are grouped together and the train and test sets maintain similar proportions of male and female individuals.

## 2. Cross-Validation (Optional)

If cross-validation folds are required, the function `group_stratified_kfold` assigns a `Fold` column to the training set. It uses scikit-learn’s `StratifiedGroupKFold` so that

- the same individual never appears in multiple folds, and
- each fold has a balanced distribution of sexes.

This step is triggered inside `SplitWrapper.split_all()` via the private helper `_make_folds`. If the project does not need cross-validation, this step can be skipped by ignoring or removing the `Fold` column.

## 3. Summary Information

For each generated split the wrapper prints a short summary, including the number of rows, the number of unique individuals and the counts per sex. These statistics help verify that the stratification and grouping behaved as expected.
