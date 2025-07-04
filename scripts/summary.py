#!/usr/bin/env python3
"""CLI entrypoint to create dataset summary tables and plots."""

from FIT_python.summary_wrapper import SummaryWrapper
import argparse
from pathlib import Path
from FIT_python import config


def main() -> int:
    parser = argparse.ArgumentParser(description="Create dataset summaries")
    parser.add_argument("--splits-dir", type=Path, default=config.SPLITS_DIR)
    parser.add_argument("--output-table", type=Path, default=config.RESULTS_DATA_DIR / "summary.csv")
    parser.add_argument("--fig-dir", type=Path, default=config.FIGURES_DIR / "summary")
    args = parser.parse_args()

    wrapper = SummaryWrapper()
    return wrapper.summarize_all(args.splits_dir, args.output_table, args.fig_dir)


if __name__ == "__main__":
    import sys
    sys.exit(main())
