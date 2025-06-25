#!/usr/bin/env python3
# scripts/create_splits.py

from pathlib import Path
import pandas as pd
from FIT_python.config import RAW_DIR, PROCESSED_SPLITS_DIR, normalize_dataset_name, DEBUG_MODE
from FIT_python.splits_wrapper import all_splits
from FIT_python.path_utils import split_path

def ensure_dir(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)

def save_df(df: pd.DataFrame, path: Path):
    ensure_dir(path)
    df.to_parquet(path, index=False)
    print(f"Saved {path} ({len(df)} rows)")

def main():
    raw_dir = RAW_DIR
    if not raw_dir.exists():
        msg = f"Required file not found: {raw_dir}"
        if DEBUG_MODE:
            raise FileNotFoundError(msg)
        else:
            print("Skipping:", msg)
            return
    out_dir = PROCESSED_SPLITS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    splits = all_splits(raw_dir)
    for name, parts in splits.items():
        for split_name, df in parts.items():
            save_df(df, split_path(name, split_name))

if __name__ == "__main__":
    main()
