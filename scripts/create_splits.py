#!/usr/bin/env python3
# scripts/create_splits.py

from pathlib import Path
import pandas as pd
from FIT_python.config import RAW_DIR, SPLITS_DIR, normalize_dataset_name
from FIT_python.splits_wrapper import all_splits

def ensure_dir(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)

def save_df(df: pd.DataFrame, path: Path):
    ensure_dir(path)
    df.to_parquet(path, index=False)
    print(f"Saved {path} ({len(df)} rows)")

def main():
    raw_dir = RAW_DIR
    split_dir = SPLITS_DIR

    splits = all_splits(raw_dir)
    for name, parts in splits.items():
        norm_name = normalize_dataset_name(name)
        base = split_dir / norm_name
        for split_name, df in parts.items():
            save_df(df, base / f"{split_name}.parquet")

if __name__ == "__main__":
    main()
