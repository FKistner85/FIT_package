#!/usr/bin/env python
"""Evaluate all k directories using global cutoffs."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from FIT_python.pipeline_individual_id.sequential_holdout import (
    compute_global_cutoffs,
    evaluate_with_cutoff,
)


def evaluate_k_dir(result_dir: Path) -> None:
    """Compute evaluation tables for one k directory."""
    all_csv = result_dir / "all_splits.csv"
    if not all_csv.exists():
        raise FileNotFoundError(all_csv)

    cutoffs = compute_global_cutoffs(all_csv)

    eval_mean = evaluate_with_cutoff(result_dir, cutoff=cutoffs["mean_cutoff"])
    eval_median = evaluate_with_cutoff(result_dir, cutoff=cutoffs["median_cutoff"])
    eval_low = evaluate_with_cutoff(result_dir, cutoff=cutoffs["mean_low"])
    eval_high = evaluate_with_cutoff(result_dir, cutoff=cutoffs["mean_high"])

    eval_mean.to_csv(result_dir / "eval_mean.csv", index=False)
    eval_median.to_csv(result_dir / "eval_median.csv", index=False)
    eval_low.to_csv(result_dir / "eval_mean_low.csv", index=False)
    eval_high.to_csv(result_dir / "eval_mean_high.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "exp_dir",
        type=Path,
        help="Experiment directory containing k<value> subdirectories",
    )
    parser.add_argument(
        "--k",
        type=int,
        nargs="*",
        help="Specific k values to evaluate. Default is all k* directories in exp_dir",
    )

    args = parser.parse_args()
    k_values = args.k
    base_dir: Path = args.exp_dir

    if k_values is None:
        k_values = [int(p.name[1:]) for p in base_dir.glob("k*") if p.is_dir() and p.name[1:].isdigit()]
        k_values.sort()

    for k in k_values:
        dir_path = base_dir / f"k{k}"
        if not dir_path.exists():
            print(f"{dir_path} does not exist, skipping")
            continue
        print(f"Evaluating {dir_path}")
        evaluate_k_dir(dir_path)


if __name__ == "__main__":
    main()
