#!/usr/bin/env python3
# scripts/scale_splits.py
# Uses DEBUG_MODE from config to fail-fast on missing files

"""Scale all train/test splits using configured scalers."""

from pathlib import Path
import pandas as pd
from sklearn.preprocessing import StandardScaler

from FIT_python.config import (
    PROCESSED_SPLITS_DIR,
    SCALED_DIR,
    DEFAULT_TARGETS,
    OTTER_META_COLS,
    DEBUG_MODE,
)
from FIT_python.path_utils import split_path, scaled_path

def main():
    if not PROCESSED_SPLITS_DIR.exists():
        msg = f"Required file not found: {PROCESSED_SPLITS_DIR}"
        if DEBUG_MODE:
            raise FileNotFoundError(msg)
        else:
            print("Skipping:", msg)
            return

    SCALED_DIR.mkdir(parents=True, exist_ok=True)

    datasets = set(p.stem.rsplit("_", 1)[0] for p in PROCESSED_SPLITS_DIR.glob("*_train.parquet"))

    for dataset in datasets:
        train_p = split_path(dataset, "train")
        test_p = split_path(dataset, "test")

        if not train_p.exists() or not test_p.exists():
            msg = f"Required file not found: {train_p if not train_p.exists() else test_p}"
            if DEBUG_MODE:
                raise FileNotFoundError(msg)
            else:
                print("Skipping:", msg)
                continue

        df_train = pd.read_parquet(train_p)
        df_test = pd.read_parquet(test_p)

        meta_cols = OTTER_META_COLS if "otter" in dataset.lower() else ["id"]
        feature_cols = [c for c in df_train.columns if c not in meta_cols + DEFAULT_TARGETS]

        scaler = StandardScaler()
        df_train.loc[:, feature_cols] = scaler.fit_transform(df_train[feature_cols])
        df_test.loc[:, feature_cols] = scaler.transform(df_test[feature_cols])

        df_train.to_parquet(scaled_path(dataset, "train"), index=False)
        df_test.to_parquet(scaled_path(dataset, "test"), index=False)
        print(f"Scaled dataset {dataset}")


if __name__ == "__main__":
    main()
