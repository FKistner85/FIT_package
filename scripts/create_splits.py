#!/usr/bin/env python3
# scripts/create_splits.py

import pandas as pd
from pathlib import Path
from FIT_python.data_loader import load_csv
from FIT_python.split_otter import create_train_test_split
from FIT_python.grouped_splits import train_test_group_split, group_stratified_kfold

RAW_DIR   = Path("data/raw")
SPLIT_DIR = Path("data/splits")

def ensure_dir(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)

def save_df(df: pd.DataFrame, path: Path):
    ensure_dir(path)
    df.to_parquet(path, index=False)
    print(f" -> saved {path} ({len(df)} rows)")

def main():
    for csvfile in RAW_DIR.glob("*.csv"):
        name = csvfile.stem.replace(" ", "_")
        df   = load_csv(csvfile)

        out_folder = SPLIT_DIR / name
        train_path = out_folder / "train.parquet"
        test_path  = out_folder / "test.parquet"
        folds_path = out_folder / "folds.parquet"
        inf_path   = out_folder / "inference.parquet"

        print(f"Processing {name}...")
        if "otter" in name.lower():
            # Otter-specific split: train, test, inference
            train_df, test_df, inf_df = create_train_test_split(df)
            save_df(train_df, train_path)
            save_df(test_df,  test_path)
            save_df(inf_df,  inf_path)

            # Simple group-only fold assignment on train set
            folds_df, _ = train_test_group_split(train_df)
            save_df(folds_df, folds_path)

        else:
            # Generic stratified group split
            train_df, test_df = train_test_group_split(df)
            save_df(train_df, train_path)
            save_df(test_df,  test_path)

            # StratifiedGroupKFold on train set
            folds_df = group_stratified_kfold(train_df)
            save_df(folds_df, folds_path)

if __name__ == "__main__":
    main()
