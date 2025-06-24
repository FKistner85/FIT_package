# src/FIT_python/splits_wrapper.py

from pathlib import Path
from typing import Dict
import pandas as pd

from FIT_python.data_import_utils import load_raw_files
from FIT_python.split_utils import (
    train_test_group_split,
    group_stratified_kfold,
    create_train_test_split_otter
)

def all_splits(raw_dir: Path) -> Dict[str, Dict[str, pd.DataFrame]]:
    """
    For each dataset in raw_dir:
      - if 'otter' in name: use create_train_test_split_otter
      - else: use train_test_group_split + group_stratified_kfold
    Returns mapping dataset_name -> {train, test[, inference], folds}.
    """
    dfs = load_raw_files(raw_dir)
    results: Dict[str, Dict[str, pd.DataFrame]] = {}

    for name, df in dfs.items():
        key = name.lower()
        out: Dict[str, pd.DataFrame] = {}

        if "otter" in key:
            train_df, test_df, inf_df = create_train_test_split_otter(df)
            out["train"]     = train_df
            out["test"]      = test_df
            out["inference"] = inf_df
            folds_df, _ = train_test_group_split(train_df)
            out["folds"] = folds_df

        else:
            train_df, test_df = train_test_group_split(df)
            out["train"] = train_df
            out["test"]  = test_df
            folds_df = group_stratified_kfold(train_df)
            out["folds"] = folds_df

        results[name] = out

    return results
