#!/usr/bin/env python3
# scripts/create_splits.py

from pathlib import Path
import pandas as pd
from FIT_python.splits_wrapper import all_splits

def ensure_dir(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)

def save_df(df: pd.DataFrame, path: Path):
    ensure_dir(path)
    df.to_parquet(path, index=False)
    print(f"Saved {path} ({len(df)} rows)")

def main():
    project_root = Path(__file__).resolve().parent.parent
    raw_dir  = project_root / "data" / "raw"
    split_dir= project_root / "data" / "splits"

    splits = all_splits(raw_dir)
    for name, parts in splits.items():
        base = split_dir / name.replace(" ", "_")
        for split_name, df in parts.items():
            save_df(df, base / f"{split_name}.parquet")

if __name__ == "__main__":
    main()
