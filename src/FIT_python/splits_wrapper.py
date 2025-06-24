# src/FIT_python/splits_wrapper.py

from pathlib import Path
from typing import Dict
import pandas as pd

from FIT_python.data_import_utils import load_raw_files
from FIT_python.split_utils import train_test_group_split
from FIT_python.config import DEBUG_MODE

def all_splits(raw_dir: Path) -> Dict[str, Dict[str, pd.DataFrame]]:
    """Create simple train/test splits for each dataset under ``raw_dir``."""
    dfs = load_raw_files(raw_dir)
    results: Dict[str, Dict[str, pd.DataFrame]] = {}

    for name, df in dfs.items():
        try:
            train_df, test_df = train_test_group_split(df)
        except Exception as e:
            msg = f"Splitting failed for {name}: {e}"
            if DEBUG_MODE:
                raise ValueError(msg)
            else:
                print("Skipping:", msg)
                continue
        results[name] = {"train": train_df, "test": test_df}

    return results
