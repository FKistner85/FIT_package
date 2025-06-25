# src/FIT_python/summary_wrapper.py
"""High level orchestration for dataset split summaries."""

from __future__ import annotations

from pathlib import Path
import pandas as pd

from FIT_python.summary_utils import (
    discover_splits,
    load_split_data,
    compute_summary,
    plot_summary_table,
)
import FIT_python.config as config
import sys


ALLOWED_ORIGINS = {"train", "test", "inference"}


def run_summary(
    splits_dir: Path,
    output_table: Path,
    fig_dir: Path,
    *,
    force: bool = False,
    plot: bool = True,
) -> None:
    """Create CSV and plots summarising dataset splits."""

    if not splits_dir.exists():
        raise RuntimeError(f"splits_dir does not exist: {splits_dir}")

    splits = discover_splits(splits_dir)
    if not splits:
        raise RuntimeError(f"No split files found in {splits_dir}")

    rows = []
    for dataset, parts in splits.items():
        for origin, path in parts.items():
            if origin not in ALLOWED_ORIGINS:
                raise RuntimeError(f"Unexpected origin '{origin}' in {path}")
            df = load_split_data(path)
            rows.append(compute_summary(df, dataset, origin))

    df_summary = pd.concat(rows, ignore_index=True)

    if output_table.exists() and not force:
        print(f"Skipping: {output_table} exists. Use --force to overwrite.")
    else:
        output_table.parent.mkdir(parents=True, exist_ok=True)
        df_summary.to_csv(output_table, index=False)
        print(f"Saved summary table to {output_table}")

    if plot:
        plot_summary_table(df_summary, fig_dir)


class SummaryWrapper:
    """Wrapper providing default configuration for dataset summaries."""

    def __init__(self) -> None:
        pass

    def summarize_all(
        self,
        splits_dir: Path | None = None,
        output_table: Path | None = None,
        fig_dir: Path | None = None,
    ) -> int:
        try:
            run_summary(
                splits_dir or config.SPLITS_DIR,
                output_table or config.RESULTS_DATA_DIR / "summary.csv",
                fig_dir or config.FIGURES_DIR / "summary",
                force=True,
                plot=True,
            )
            print(
                f"[SUCCESS] summary written to {config.RESULTS_DATA_DIR / 'summary.csv'}"
            )
            return 0
        except Exception as exc:
            print(f"[ERROR] {exc}", file=sys.stderr)
            if config.DEBUG_MODE:
                raise
            return 1
