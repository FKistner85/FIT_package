#!/usr/bin/env python3
# scripts/create_feature_selection.py
# Uses DEBUG_MODE from config to fail-fast on missing files

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
        X = np.load(X_path, allow_pickle=True)
        y_raw = np.load(y_path, allow_pickle=True)
        y_series = (
            pd.Series(y_raw)
            .fillna("unknown")
            .astype(str)
            .str.strip()
            .str.lower()
            .map({
                "f": "female",
                "female": "female",
                "m": "male",
                "male": "male",
            })
            .fillna("unknown")
            .map({"female": 0, "male": 1, "unknown": 2})
        )

        # Reconstruct DataFrame for column names
        # locate matching processed dataset folder (case-insensitive)
        proc_match = None
        for d in PROCESSED_DIR.iterdir():
            if d.is_dir() and normalize_dataset_name(d.name) == dataset and d.name != "numeric":
                if (d / "train.parquet").exists():
                    proc_match = d
                    break
        if proc_match is None:
            msg = f"No processed folder for {dataset} under {PROCESSED_DIR}"
            if DEBUG_MODE:
                raise FileNotFoundError(msg)
            else:
                print("Skipping:", msg)
                skipped.append(dataset)
                continue
        df_raw = pd.read_parquet(proc_match / "train.parquet")
        if "otter" in dataset.lower():
            meta = OTTER_META_COLS
        else:
            meta = ["id"]
        feature_cols = [c for c in df_raw.columns if c not in meta + DEFAULT_TARGETS]

        if y_series.nunique() < 2:
            print(f"Skipping {dataset}: only one class present")
            skipped.append(dataset)
            continue

        X_df = pd.DataFrame(X, columns=feature_cols)

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
