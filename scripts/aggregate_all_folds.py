#!/usr/bin/env python
"""Command-line wrapper for :func:`aggregate_all_folds`."""
from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path

from FIT_python.utils import aggregate_all_folds


def main() -> None:
    parser = ArgumentParser(description="Combine 'all_folds.csv' files across species")
    parser.add_argument("exp_dir", type=Path, help="Experiment directory with species subfolders")
    parser.add_argument("-o", "--output", type=Path, default=None, help="Output CSV path")
    args = parser.parse_args()

    df = aggregate_all_folds(args.exp_dir, args.output)
    print(f"[INFO] combined {len(df)} rows")


if __name__ == "__main__":
    main()
