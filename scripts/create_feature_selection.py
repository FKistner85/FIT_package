#!/usr/bin/env python3
"""Simple feature selection on scaled datasets."""

from pathlib import Path
import pandas as pd

from FIT_python.config import (
    SCALED_DIR,
    FEATURE_SELECTED_DIR,
    DEFAULT_TARGETS,
    OTTER_META_COLS,
    DEBUG_MODE,
)
from FIT_python.path_utils import scaled_path, feature_selected_path


N_FEATURES = 2  # select top N features by variance


def main() -> None:
    if not SCALED_DIR.exists():
        msg = f"Required file not found: {SCALED_DIR}"
        if DEBUG_MODE:
            raise FileNotFoundError(msg)
        else:
            print("Skipping:", msg)
            return

    FEATURE_SELECTED_DIR.mkdir(parents=True, exist_ok=True)
    datasets = set(p.stem.rsplit("_", 1)[0] for p in SCALED_DIR.glob("*_train.parquet"))

    for dataset in datasets:
        train_p = scaled_path(dataset, "train")
        test_p = scaled_path(dataset, "test")
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

        variances = df_train[feature_cols].var().sort_values(ascending=False)
        selected = variances.head(N_FEATURES).index.tolist()

        for split_name, df in [("train", df_train), ("test", df_test)]:
            df_sel = df[meta_cols + DEFAULT_TARGETS + selected]
            df_sel.to_parquet(feature_selected_path(dataset, split_name), index=False)
        print(f"Selected features for {dataset}: {selected}")


if __name__ == "__main__":
    main()
