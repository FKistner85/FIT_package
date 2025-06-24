#!/usr/bin/env python3
# scripts/create_feature_selection.py
# Uses DEBUG_MODE from config to fail-fast on missing files

"""
Script to run feature selection on all numeric datasets with result caching.
"""

from pathlib import Path
import json
import pandas as pd
import numpy as np

from FIT_python.config import (
    NUMERIC_DIR,
    PROCESSED_DIR,
    DEFAULT_TARGETS,
    OTTER_META_COLS,
    FS_DEFAULT_METHODS,
    FS_TARGET_FEATURE_COUNTS,
    normalize_dataset_name,
    DEBUG_MODE,
)
from FIT_python.feature_utils import run_feature_selection_methods

def main():
    successes = []
    skipped = []
    for ds_folder in NUMERIC_DIR.iterdir():
        if not ds_folder.is_dir():
            continue
        dataset = normalize_dataset_name(ds_folder.name)
        print(f"\n=== Feature selection for {dataset} ===")

        out_dir = PROCESSED_DIR / dataset / "feature_selection"
        out_dir.mkdir(parents=True, exist_ok=True)

        X_path = ds_folder / "X_train.npy"
        y_path = ds_folder / "y_train.npy"
        missing = None
        for path in [X_path, y_path]:
            if not path.exists():
                missing = path
                break
        if missing:
            msg = f"Required file not found: {missing}"
            if DEBUG_MODE:
                raise FileNotFoundError(msg)
            else:
                print("Skipping:", msg)
                skipped.append(dataset)
                continue
        X = np.load(X_path)
        y = np.load(y_path)

        # Reconstruct DataFrame for column names
        df_raw = pd.read_parquet(PROCESSED_DIR / dataset / "train.parquet")
        if "otter" in dataset.lower():
            meta = OTTER_META_COLS
        else:
            meta = ["id"]
        feature_cols = [c for c in df_raw.columns if c not in meta + DEFAULT_TARGETS]
        X_df = pd.DataFrame(X, columns=feature_cols)

        # Always convert y to a 1D label series
        mapping_path = ds_folder / "target_mapping.json"
        if mapping_path.exists():
            with open(mapping_path, "r", encoding="utf-8") as f:
                mapping = json.load(f)
            # columns corresponding to the 'sex' target
            oh_cols = mapping.get("sex", [])
            start_idx = 0
            for col in DEFAULT_TARGETS:
                if col == "sex":
                    break
                start_idx += len(mapping.get(col, []))
            if y.ndim > 1:
                y_slice = y[:, start_idx : start_idx + len(oh_cols)]
                idx = np.argmax(y_slice, axis=1)
                labels = [oh_cols[i].split("_")[-1] for i in idx]
                y_series = pd.Series(labels, name="sex")
            else:
                y_series = pd.Series(y, name="sex")
        else:
            if y.ndim > 1 and DEBUG_MODE:
                raise FileNotFoundError(f"Missing mapping file: {mapping_path}")
            # Fall back to simple series
            y_series = pd.Series(y.ravel(), name="sex")

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

        successes.append(dataset)

    print("\nFeature selection summary:")
    print("  Successful:", successes)
    if skipped:
        print("  Skipped:", skipped)

if __name__ == "__main__":
    main()
