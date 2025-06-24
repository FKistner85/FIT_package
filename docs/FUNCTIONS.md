# Function Reference

This document describes the key functions provided by the FIT package. The order roughly follows the sequence used in `tests/test_all_scripts.ipynb` which runs three data scripts in succession.

## 1. Data Loading (`dataload.py`)

### `load_raw_files(folder: Path, add_id: bool = True, id_prefix: Optional[str] = None) -> Dict[str, pd.DataFrame>`
Loads every CSV or Excel file from a directory, cleans column names, normalises the individual identifier column and optionally adds an `id` column. Returns a dictionary mapping the file stem to each loaded `DataFrame`.

### `main()` (in `scripts/dataload.py`)
Entrypoint for the data loading script. It calls `load_raw_files` using the path configured in `FIT_python.config.RAW_DIR` and prints the shape and a preview of each loaded `DataFrame`.

## 2. Split Creation (`create_splits.py`)

### `ensure_dir(path: Path)`
Create the parent directory for a given path if it does not already exist.

### `save_df(df: pd.DataFrame, path: Path)`
Helper used by the script to save a `DataFrame` to a parquet file.

### `all_splits(raw_dir: Path) -> Dict[str, Dict[str, pd.DataFrame]]`
Function from `FIT_python.splits_wrapper`. For every dataset under `raw_dir` it creates train/test splits (and folds and inference data for the otter dataset). Internally it uses `train_test_group_split`, `group_stratified_kfold` and `create_train_test_split_otter` from `FIT_python.split_utils`.

### `train_test_group_split(df: pd.DataFrame, ...) -> Tuple[pd.DataFrame, pd.DataFrame]`
Group aware train/test split ensuring that no individual appears in both sets and that the `sex` distribution is preserved.

### `group_stratified_kfold(df: pd.DataFrame, ...) -> pd.DataFrame`
Assigns a `Fold` column to the data using `StratifiedGroupKFold` so that cross validation keeps individuals together and respects the `sex` stratification.

### `create_train_test_split_otter(df: pd.DataFrame, seed: int = GLOBAL_RANDOM_SEED)`
Deterministically constructs train, test and inference splits for the otter dataset.

### `main()` (in `scripts/create_splits.py`)
Runs `all_splits` on the raw data directory and saves all resulting dataframes to `data/splits`.

## 3. Landmark Mapping (`create_landmark_map.py`)

### `main()` (in `scripts/create_landmark_map.py`)
Loads the Eurasian Otter dataset using `load_raw_files`, analyses the feature columns and writes `otter_landmark_map.json` and `otter_point_map.json` which describe which landmarks each feature belongs to.

## 4. Additional Utilities

### `clean_columns(columns: Union[List[str], pd.Index]) -> List[str>`
Lowercase column names, strip spaces and replace non-alphanumeric characters with underscores.

### `load_csv(path: Path) -> pd.DataFrame`
Read a CSV file into a DataFrame.

### `load_excel(path: Path) -> pd.DataFrame`
Read an Excel file into a DataFrame.

### `sanitize_labels(df: pd.DataFrame, target_cols: List[str], skip_fillna: Optional[List[str]] = None, mapping: Optional[Dict[str, str]] = None) -> pd.DataFrame`
Clean text labels in the specified columns, optionally applying mappings and replacing missing values with `'unknown'`.

### `DataImporter` class (`data_import_wrapper.py`)
Convenience wrapper that loads raw files from a directory and optionally cleans specified target columns. Methods:
- `load()` – load raw files via `load_raw_files`.
- `clean()` – apply `sanitize_labels` to each data frame.
- `run()` – perform load and clean in one call.

### `DataPipeline` class (`pipeline.py`)
Pipeline for loading a single file or a directory of raw data, cleaning target columns and splitting into meta, target and feature data. Methods:
- `load()`
- `clean_targets()`
- `split()`
- `run()`

### `all_splits` in `FIT_python.splits`
Simpler variant that either calls `create_train_test_split_otter` or `train_test_group_split` depending on the dataset name and returns the resulting splits.

### Functions in `grouped_splits.py`
`train_test_group_split` and `group_stratified_kfold` provide the same logic as in `split_utils.py` but without additional helpers.

---
These functions together support loading raw data, generating machine-learning ready splits and producing landmark mapping files. They are invoked in the above order when executing `tests/test_all_scripts.ipynb`.
