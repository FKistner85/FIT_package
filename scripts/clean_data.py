#!/usr/bin/env python3
"""Simple cleaning step copying raw CSV files to Parquet."""
import pandas as pd
from FIT_python.config import RAW_DIR, CLEANED_DIR, DEBUG_MODE
from FIT_python.data_import_utils import load_raw_files


def main() -> None:
    if not RAW_DIR.exists():
        msg = f"Required file not found: {RAW_DIR}"
        if DEBUG_MODE:
            raise FileNotFoundError(msg)
        else:
            print("Skipping:", msg)
            return
    CLEANED_DIR.mkdir(parents=True, exist_ok=True)
    dfs = load_raw_files(RAW_DIR)
    for name, df in dfs.items():
        out = CLEANED_DIR / f"{name}.parquet"
        df.to_parquet(out, index=False)
        print(f"Cleaned {name} -> {out}")


if __name__ == "__main__":
    main()
