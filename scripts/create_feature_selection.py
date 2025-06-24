#!/usr/bin/env python3
# scripts/create_feature_selection.py

"""
Script to run feature selection on all numeric datasets with result caching.
"""

from pathlib import Path
import pandas as pd
import numpy as np

from FIT_python.config import (
    NUMERIC_DIR,
    PROCESSED_DIR,
    DEFAULT_TARGETS,
    OTTER_META_COLS,
    FS_DEFAULT_METHODS,
    FS_TARGET_FEATURE_COUNTS
)
from FIT_python.feature_utils import run_feature_selection_methods

def main():
    for ds_folder in NUMERIC_DIR.iterdir():
        if not ds_folder.is_dir():
            continue
        dataset = ds_folder.name
        print(f"\n=== Feature selection for {dataset} ===")

        out_dir = PROCESSED_DIR / dataset / "feature_selection"
        out_dir.mkdir(parents=True, exist_ok=True)

        # Load numeric data
        X = np.load(ds_folder / "X_train.npy")
        y = np.load(ds_folder / "y_train.npy")

        # Reconstruct DataFrame for column names
        df_raw = pd.read_parquet(PROCESSED_DIR / dataset / "train.parquet")
        if "otter" in dataset.lower():
            meta = OTTER_META_COLS
        else:
            meta = ["id"]
        feature_cols = [c for c in df_raw.columns if c not in meta + DEFAULT_TARGETS]
        X_df = pd.DataFrame(X, columns=feature_cols)
        y_series = pd.Series(y, name="target")

        # Iterate through methods
        for method in FS_DEFAULT_METHODS:
            # Determine counts to use
            if method.startswith("forward"):
                counts = [None] + FS_TARGET_FEATURE_COUNTS
            else:
                counts = FS_TARGET_FEATURE_COUNTS

            for cnt in counts:
                # Build result filename
                if method.startswith("forward") and cnt is None:
                    filename = f"{method}_p.txt"
                else:
                    filename = f"{method}_{cnt}f.txt"
                filepath = out_dir / filename

                # Skip if already exists
                if filepath.exists():
                    print(f"Skipping {filename}, already exists.")
                    continue

                # Run selection
                print(f"Running {method} with count={cnt}")
                results = run_feature_selection_methods(
                    X_df, y_series,
                    methods=[method],
                    target_feature_counts=cnt if cnt is not None else None,
                    verbose=False
                )
                # Extract the sole entry
                key = next(iter(results))
                feats = results[key]

                # Save to txt
                pd.Series(feats).to_csv(filepath, index=False, header=False)
                print(f"Saved {filename} ({len(feats)} features)")

if __name__ == "__main__":
    main()
