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
from tqdm.auto import tqdm
from FIT_python.utils import get_species_paths


def run_summary(
    splits_dir: Path,
    species: str,
    *,
    force: bool = False,
    plot: bool = True,
) -> None:
    """Create Parquet and plots summarising each split for ``species``."""
    paths = get_species_paths(species, section="dataprocessing")
    output_table = paths["tables"] / "summary.parquet"
    fig_dir = paths["figures"] / "summary"

    splits = discover_splits(splits_dir)
    if not splits:
        raise RuntimeError(f"No split files found in {splits_dir}")

    rows: list[pd.DataFrame] = []

    for dataset, parts in tqdm(splits.items(), desc="Datasets"):
        for origin, path in tqdm(parts.items(), desc="Splits", leave=False):
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
        df_summary.to_parquet(output_table, index=False, compression="gzip")
        print(f"[SUCCESS] summary written to {output_table}")

    if plot:
        plot_summary_table(df_summary, fig_dir)
        plot_split_proportions(df_summary, fig_dir)


class SummaryWrapper:
    def summarize_all(
        self,
        splits_dir: Path | None = None,
    ) -> int:
        try:
            base_dir = splits_dir or config.SPLITS_DIR
            for species_dir in sorted(base_dir.iterdir()):
                if species_dir.is_dir():
                    run_summary(species_dir, species_dir.name, force=True, plot=True)
            return 0
        except Exception as exc:
            print(f"[ERROR] {exc}", file=sys.stderr)
