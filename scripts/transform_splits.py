#!/usr/bin/env python3
"""Transform all split datasets to NumPy format.

DEBUG_MODE from :mod:`FIT_python.config` controls fail-fast behaviour. If
enabled, missing Parquet files raise an exception. Otherwise the dataset is
skipped with a warning.
"""

import sys
from FIT_python.transform_wrapper import TransformWrapper
from FIT_python.config import DEBUG_MODE, SPLITS_DIR

def main() -> int:
    wrapper = TransformWrapper()
    print(f"[INFO] Starting transformation of all splits in {SPLITS_DIR}")

    if not SPLITS_DIR.exists():
        msg = f"Required directory not found: {SPLITS_DIR}"
        if DEBUG_MODE:
            print(f"[ERROR] {msg}")
            raise FileNotFoundError(msg)
        else:
            print(f"[WARNING] {msg} — skipping.")
            return 1

    for folder in SPLITS_DIR.iterdir():
        if not folder.is_dir():
            continue

        name = folder.name.replace(" ", "_")
        print(f"\n[INFO] === Dataset: {name} ===")

        train_pq = folder / "train.parquet"
        test_pq  = folder / "test.parquet"

        # Check for missing files
        missing = None
        for path in (train_pq, test_pq):
            if not path.exists():
                missing = path
                break

        if missing:
            msg = f"Required file not found: {missing}"
            if DEBUG_MODE:
                print(f"[ERROR] {msg}")
                raise FileNotFoundError(msg)
            else:
                print(f"[WARNING] {msg} — skipping dataset {name}.")
                continue

        # Perform transform
        print(f"[INFO] Calling transform_dataset('{name}', 'train.parquet', 'test.parquet')")
        try:
            wrapper.transform_dataset(name, train_pq, test_pq)
            print(f"[SUCCESS] Dataset '{name}' transformed successfully.")
        except Exception as exc:
            print(f"[ERROR] Failed to transform '{name}': {exc}", file=sys.stderr)
            if DEBUG_MODE:
                raise
            else:
                print(f"[WARNING] Skipping '{name}' due to error.")
                continue

    print("\n[INFO] All datasets processed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())