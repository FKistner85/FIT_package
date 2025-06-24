# src/FIT_python/splits.py

from pathlib import Path
import pandas as pd
from typing import Dict
from FIT_python.data_loader import load_raw_files
from FIT_python.grouped_splits import train_test_group_split
from FIT_python.split_otter import create_train_test_split

def all_splits(raw_dir: Path) -> Dict[str, Dict[str, pd.DataFrame]]:
    """
    For each dataset in raw_dir:
      - if 'otter' in name: use create_train_test_split (produces train, test, inference)
      - else: use generic train_test_group_split (produces train, test)
    Returns a dict mapping dataset_name -> dict of DataFrames.
    """
    dfs = load_raw_files(raw_dir)
    splits = {}
    for name, df in dfs.items():
        key = name.lower()
        if 'otter' in key:
            train_df, test_df, inference_df = create_train_test_split(df)
            splits[name] = {
                'train': train_df,
                'test': test_df,
                'inference': inference_df
            }
        else:
            train_df, test_df = train_test_group_split(df)
            splits[name] = {
                'train': train_df,
                'test': test_df
            }
    return splits

if __name__ == "__main__":
    # Example usage
    raw_dir = Path("data/raw")
    splits = all_splits(raw_dir)
    for name, parts in splits.items():
        print(f"Dataset: {name}")
        for split_name, df in parts.items():
            print(f"  {split_name} size:", len(df))
