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
    try:
        if not SPLITS_DIR.exists():
            msg = f"Required file not found: {SPLITS_DIR}"
            if DEBUG_MODE:
                raise FileNotFoundError(msg)
            else:
                print("Skipping:", msg)
                return 1
        for folder in SPLITS_DIR.iterdir():
            if not folder.is_dir():
                continue
            name = folder.name.replace(" ", "_")
            train_pq = folder / "train.parquet"
            test_pq = folder / "test.parquet"

            missing = None
            for path in (train_pq, test_pq):
                if not path.exists():
                    missing = path
                    break
            if missing:
                msg = f"Required file not found: {missing}"
                if DEBUG_MODE:
                    raise FileNotFoundError(msg)
                else:
                    print("Skipping dataset:", msg)
                    continue

            wrapper.transform_dataset(name, train_pq, test_pq)
    except Exception as exc:
        print(f"Error: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

