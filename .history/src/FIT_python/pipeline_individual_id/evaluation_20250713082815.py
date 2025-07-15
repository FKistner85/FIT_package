from __future__ import annotations

"""Utility helpers for evaluating individual ID pipelines."""

from collections import Counter
from typing import Iterable, Mapping

import FIT_python.config as config

import pandas as pd
from sklearn.metrics import confusion_matrix


def compute_confusion(
    results_df: pd.DataFrame,
    *,
    true_col: str = "same_individual",
    pred_col: str = "pred",
) -> pd.DataFrame:
    """Return a confusion matrix for ``results_df``.

    Parameters
    ----------
    results_df:
        DataFrame containing ground-truth and prediction columns.
    true_col:
        Column name with the true ``same_individual`` labels.
    pred_col:
        Column name with the predicted labels.

    The function maps string labels ``"True"``/``"False"`` to ``1``/``0`` and
    returns a ``pandas.DataFrame`` with labelled rows/columns.
    """
    if true_col not in results_df.columns or pred_col not in results_df.columns:
        raise KeyError(f"Missing '{true_col}' or '{pred_col}' in results_df")

    mapping = {"True": 1, "False": 0, True: 1, False: 0, 1: 1, 0: 0}
    raw_true = results_df[true_col]
    raw_pred = results_df[pred_col]

    if config.DEBUG_MODE:
        print(
            f"[DEBUG] unique true labels: {sorted(raw_true.dropna().unique())};"
            f" pred labels: {sorted(raw_pred.dropna().unique())}"
        )

    y_true = raw_true.map(mapping)
    y_pred = raw_pred.map(mapping)

    if y_true.isna().any() or y_pred.isna().any():
        invalid = y_true.isna() | y_pred.isna()
        if config.DEBUG_MODE:
            print("[DEBUG] dropping rows with NaN after mapping:\n", results_df[invalid])
        y_true = y_true[~invalid]
        y_pred = y_pred[~invalid]

    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)

    cm = confusion_matrix(y_true, y_pred, labels=[1, 0])
    return pd.DataFrame(
        cm, index=["true_same", "true_diff"], columns=["pred_same", "pred_diff"]
    )


def report_skipped(skipped_counts: Mapping[str, int]) -> str:
    """Return a multi-line summary of skipped validation counts."""
    total = sum(skipped_counts.values())
    lines = [f"{name}: {cnt}" for name, cnt in skipped_counts.items()]
    lines.append(f"Total: {total}")
    return "\n".join(lines)


def average_trail_stats(summary_tables: Iterable[pd.DataFrame]) -> pd.Series:
    """Return the column-wise mean of ``summary_tables``.

    The function expects each table to contain a ``sub_size == 'Total'`` row.
    Only numeric columns of these rows are averaged.
    """
    totals = [df[df["sub_size"] == "Total"] for df in summary_tables]
    if not totals:
        raise ValueError("No summary tables provided")

    concat = pd.concat(totals, ignore_index=True)
    numeric = concat.select_dtypes("number")
    return numeric.mean()
