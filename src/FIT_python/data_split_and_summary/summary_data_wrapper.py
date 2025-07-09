# src/FIT_python/step_02_b_summary_datasets/wrapper.py

from __future__ import annotations
from pathlib import Path
import pandas as pd
import sys

from FIT_python.data_split_and_summary.summary_data_utils import (
    discover_splits,
    load_split_data,
    compute_summary,
    plot_summary_table,
    plot_split_proportions,
)
import FIT_python.config as config

def run_summary(
    splits_dir: Path,
    output_table: Path,
    fig_dir: Path,
    *,
    force: bool = False,
    plot: bool = True,
) -> None:
    """Create CSV and plots summarising each split separately."""
    splits = discover_splits(splits_dir)
    if not splits:
        raise RuntimeError(f"No split files found in {splits_dir}")

    rows: list[pd.DataFrame] = []

    for dataset, parts in splits.items():
        for origin, path in parts.items():
            df = load_split_data(path)
            # skip empty splits entirely
            summary_df = compute_summary(df, dataset, origin)
            if not summary_df.empty:
                rows.append(summary_df)

    df_summary = pd.concat(rows, ignore_index=True)
    if output_table.exists() and not force:
        print(f"Skipping: {output_table} exists. Use --force to overwrite.")
    else:
        output_table.parent.mkdir(parents=True, exist_ok=True)
        df_summary.to_csv(output_table, index=False)
        print(f"[SUCCESS] summary written to {output_table}")

    if plot:
        plot_summary_table(df_summary, fig_dir)
        plot_split_proportions(df_summary, fig_dir)

class SummaryWrapper:
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
            return 0
        except Exception as exc:
            print(f"[ERROR] {exc}", file=sys.stderr)
