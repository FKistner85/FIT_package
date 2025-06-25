#!/usr/bin/env python3
"""CLI entrypoint to create dataset summary tables and plots."""

from __future__ import annotations

import argparse
from pathlib import Path

from FIT_python.config import SPLITS_DIR, FIGURES_DIR, RESULTS_DATA_DIR
from FIT_python.summary_wrapper import run_summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Create dataset summaries")
    parser.add_argument(
        "--splits-dir",
        type=Path,
        default=SPLITS_DIR,
        help="Directory containing split Parquet files",
    )
    parser.add_argument(
        "--output-table",
        type=Path,
        default=RESULTS_DATA_DIR / "summary.csv",
        help="Path for the output CSV table",
    )
    parser.add_argument(
        "--fig-dir",
        type=Path,
        default=FIGURES_DIR / "summary",
        help="Directory to store summary plots",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing output table",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip plot generation",
    )

    args = parser.parse_args()

    run_summary(
        args.splits_dir,
        args.output_table,
        args.fig_dir,
        force=args.force,
        plot=not args.no_plot,
    )


if __name__ == "__main__":
    main()
