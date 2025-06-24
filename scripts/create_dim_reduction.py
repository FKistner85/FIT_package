#!/usr/bin/env python3
"""Run dimensionality reduction on all processed datasets."""

from pathlib import Path
import pandas as pd
from FIT_python.config import (
    PROCESSED_DIR,
    DEFAULT_TARGETS,
    OTTER_META_COLS,
    DEBUG_MODE,
)
from FIT_python.dim_reduction_wrapper import reduce_all


def main():
    methods = ["pca", "tsne", "umap"]
    n_components = 2

    X_dict = {}
    for ds_folder in PROCESSED_DIR.iterdir():
        if not ds_folder.is_dir() or ds_folder.name == "numeric":
            continue
        dataset = ds_folder.name
        train_path = ds_folder / "train.parquet"
        if not train_path.exists():
            msg = f"Required file not found: {train_path}"
            if DEBUG_MODE:
                raise FileNotFoundError(msg)
            else:
                print("Skipping:", msg)
                continue
        df = pd.read_parquet(train_path)
        if "otter" in dataset.lower():
            meta = OTTER_META_COLS
        else:
            meta = ["id"]
        feature_cols = [c for c in df.columns if c not in meta + DEFAULT_TARGETS]
        X_dict[dataset] = df[feature_cols]

    reduce_all(X_dict, methods, n_components)


if __name__ == "__main__":
    main()
