# src/FIT_python/splits_wrapper.py

from pathlib import Path
from typing import Dict
import pandas as pd

from FIT_python.data_import_utils import load_raw_files
from FIT_python.split_utils import train_test_group_split
import FIT_python.config as config
from FIT_python.path_utils import split_path
import sys

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


class SplitsWrapper:
    """Convenience wrapper to create and store train/test splits."""

    def __init__(self) -> None:
        pass

    def split_all(self) -> int:
        if not config.RAW_DIR.exists():
            msg = f"Required directory not found: {config.RAW_DIR}"
            print(f"[ERROR] {msg}", file=sys.stderr)
            if config.DEBUG_MODE:
                raise FileNotFoundError(msg)
            return 1

        config.PROCESSED_SPLITS_DIR.mkdir(parents=True, exist_ok=True)
        splits = all_splits(config.RAW_DIR)
        for name, parts in splits.items():
            for split_name, df in parts.items():
                out = split_path(name, split_name)
                df.to_parquet(out, index=False)
                print(f"[REPORT] {name} {split_name}: shape={df.shape}")
        print("[SUCCESS] split_all completed.")
        return 0
