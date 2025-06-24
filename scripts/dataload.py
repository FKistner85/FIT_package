"""dataload.py

Load all raw data files and display their shapes and previews.
Usage: run in a notebook or as a script.
"""
from pathlib import Path
import pandas as pd
from FIT_python.data_loader import load_raw_files

def main():
    # 1) Specify the raw data directory
    raw_dir = Path('data/raw')  # Adjust path as needed

    # 2) Load all raw files
    dfs = load_raw_files(raw_dir)

    # 3) Print shape of each DataFrame
    for name, df in dfs.items():
        print(f"DataFrame '{name}' shape: {df.shape}")

    # 4) Print first 5 rows of each DataFrame
    for name, df in dfs.items():
        print(f"\nPreview of '{name}':")
        print(df.head(), "\n")

if __name__ == '__main__':
    main()
