#!/usr/bin/env python3
# scripts/scale_splits.py
# Uses DEBUG_MODE from config to fail-fast on missing files

"""Scale all train/test splits using configured scalers."""

from pathlib import Path
import pandas as pd
from sklearn.preprocessing import StandardScaler
import sys

from FIT_python.config import (
    PROCESSED_SPLITS_DIR,
    SCALED_DIR,
    DEFAULT_TARGETS,
    OTTER_META_COLS,
    DEBUG_MODE,
)
from FIT_python.path_utils import split_path, scaled_path

def main():
    print(f"[INFO] Starting scaling of all splits in {PROCESSED_SPLITS_DIR}")

    if not PROCESSED_SPLITS_DIR.exists():
        msg = f"Required directory not found: {PROCESSED_SPLITS_DIR}"
        if DEBUG_MODE:
            print(f"[ERROR] {msg}")
            raise FileNotFoundError(msg)
        else:
            print(f"[WARNING] {msg} — skipping.")
            return

    SCALED_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Ensured output directory exists: {SCALED_DIR}")

    # Find all dataset prefixes by stripping "_train.parquet"
    datasets = set(p.stem.rsplit("_", 1)[0] for p in PROCESSED_SPLITS_DIR.glob("*_train.parquet"))
    print(f"[INFO] Found datasets to scale: {sorted(datasets)}")

    for dataset in sorted(datasets):
        print(f"\n[INFO] === Scaling dataset: {dataset} ===")
        train_p = split_path(dataset, "train")
        test_p  = split_path(dataset, "test")

        # Check for missing files
        if not train_p.exists() or not test_p.exists():
            missing = train_p if not train_p.exists() else test_p
            msg = f"Required file not found: {missing}"
            if DEBUG_MODE:
                print(f"[ERROR] {msg}")
                raise FileNotFoundError(msg)
            else:
                print(f"[WARNING] {msg} — skipping dataset {dataset}.")
                continue

        print(f"[INFO] Reading train split from {train_p}")
        df_train = pd.read_parquet(train_p)
        print(f"[INFO] Reading test split from {test_p}")
        df_test  = pd.read_parquet(test_p)

        meta_cols = OTTER_META_COLS if "otter" in dataset.lower() else ["id"]
        feature_cols = [c for c in df_train.columns if c not in meta_cols + DEFAULT_TARGETS]
        print(f"[INFO] Meta columns: {meta_cols}")
        print(f"[INFO] Target columns: {DEFAULT_TARGETS}")
        print(f"[INFO] Feature columns ({len(feature_cols)}): {feature_cols}")

        scaler = StandardScaler()
        print("[INFO] Fitting scaler on training features...")
        df_train.loc[:, feature_cols] = scaler.fit_transform(df_train[feature_cols])
        print("[INFO] Transforming test features...")
        df_test.loc[:, feature_cols] = scaler.transform(df_test[feature_cols])

        out_train = scaled_path(dataset, "train")
        out_test  = scaled_path(dataset, "test")
        df_train.to_parquet(out_train, index=False)
        print(f"[SUCCESS] Wrote scaled train split to {out_train}")
        df_test.to_parquet(out_test, index=False)
        print(f"[SUCCESS] Wrote scaled test split to  {out_test}")

    print("\n[INFO] All datasets scaled.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
