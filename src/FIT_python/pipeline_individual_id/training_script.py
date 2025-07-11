from __future__ import annotations

"""Example training routine integrating evaluation utilities."""

from collections import defaultdict
from typing import Iterable

import pandas as pd

from FIT_python.pipeline_individual_id.generate_trails_and_trailpairs import (
    generate_pairwise_comparisons_from_df,
)
from FIT_python.pipeline_individual_id.pairwise_individual_id_pipeline import (
    run_embedding_once_pipeline,
)
from FIT_python.pipeline_individual_id.evaluation import (
    compute_confusion,
    report_skipped,
    average_trail_stats,
)


def train_pipelines(
    df: pd.DataFrame,
    feature_cols: Iterable[str],
    *,
    iterations: int = 1,
) -> pd.DataFrame:
    """Run the embedding pipeline and summarise results."""
    summary_tables = []
    skipped = defaultdict(int)
    all_results = []

    for it in range(iterations):
        comps, summary = generate_pairwise_comparisons_from_df(df)
        summary_tables.append(summary)

        if not comps:
            skipped["embedding"] += 1
            continue

        res = run_embedding_once_pipeline(
            comps,
            df,
            feature_cols=list(feature_cols),
        )
        df_res = pd.DataFrame(res)
        all_results.append(df_res)

        cm = compute_confusion(df_res, true_col="same_individual", pred_col="pred")
        print(f"Iteration {it} confusion:\n{cm}\n")

    print(report_skipped(skipped))
    print("Average trail statistics:")
    print(average_trail_stats(summary_tables))

    return pd.concat(all_results, ignore_index=True) if all_results else pd.DataFrame()
