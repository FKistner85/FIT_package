from __future__ import annotations
from ast import literal_eval
from typing import Iterable, List, Dict
from FIT_python.soft_config import SOFT_CONFIG

import numpy as np
from scipy.stats import chi2







"""Utility helpers for evaluating individual ID pipelines."""
from typing import Iterable, List, Dict

from collections import Counter
from typing import Iterable, Mapping
from FIT_python.soft_config import SOFT_CONFIG
import FIT_python.config as config

import pandas as pd
from sklearn.metrics import confusion_matrix


def sequential_holdout_ids(
    unique_ids: Iterable[str],
    val_sizes: Iterable[int] = tuple(
        SOFT_CONFIG["pipeline_individual_id"]["sequential_holdout_val_sizes"]
    ),
    n_iter: int = 1,
    random_state: int | None = None,
) -> List[Dict[str, List[str]]]:
    """Generate sequential train/validation splits based on unique IDs.

    For each iteration a new validation set is drawn for every entry in
    ``val_sizes``.
    Each split is therefore independent and does not build on the previous one
    within the same iteration.
    """

    ids = list(unique_ids)
    if len(ids) < 3:
        raise ValueError("Need at least three unique IDs for holdouts")

    rng = np.random.default_rng(random_state)
    results = []
    for it in range(n_iter):
        for n_val in val_sizes:
            n_val_eff = max(2, min(n_val, len(ids) - 1))
            val_ids = list(rng.choice(ids, size=n_val_eff, replace=False))
            train_ids = [i for i in ids if i not in val_ids]
            results.append(
                {
                    "iteration": it,
                    "n_val": n_val_eff,
                    "val_ids": val_ids,
                    "train_ids": train_ids,
                }
            )
    return results

def parse_list(value) -> np.ndarray:
    """Return ``value`` as a NumPy ``float`` array if possible.

    Parameters
    ----------
    value:
        Either a string representation of a list or an actual sequence
        of numeric values.

    This helper previously only accepted strings and returned an empty
    array when called with list objects.  During the sequential holdout
    evaluation the coordinate lists are still Python ``list`` instances,
    which resulted in failed parsing and consequently ``False``
    predictions.  By accepting lists and arrays directly the overlap
    logic works correctly regardless of whether results are read from a
    CSV file or computed on the fly.
    """

    if isinstance(value, (list, tuple, np.ndarray)):
        return np.asarray(value, dtype=float)

    if isinstance(value, str):
        try:
            return np.asarray(literal_eval(value), dtype=float)
        except Exception:
            return np.empty(0, dtype=float)

    return np.empty(0, dtype=float)


import numpy as np
from scipy.stats import chi2

def compute_overlap_jsl_style(row, p=0.5) -> bool:
    """Return True if the 50% confidence ellipses of A/B overlap."""
    # 1) Daten einlesen
    xa, ya = parse_list(row["coords_a_x"]), parse_list(row["coords_a_y"])
    xb, yb = parse_list(row["coords_b_x"]), parse_list(row["coords_b_y"])
    if len(xa) < 2 or len(xb) < 2 or len(xa) != len(ya) or len(xb) != len(yb):
        return False

    pts_a = np.column_stack([xa, ya])
    pts_b = np.column_stack([xb, yb])

    # 2) Zentren
    mu_a = pts_a.mean(axis=0)
    mu_b = pts_b.mean(axis=0)

    # 3) Kovarianzmatrizen
    cov_a = np.cov(pts_a, rowvar=False)
    cov_b = np.cov(pts_b, rowvar=False)

    # 4) Chi-Quadrat-Faktor für df=2
    chi_val = chi2.ppf(p, df=2)

    # 5) Eigen-Dekomposition
    vals_a, vecs_a = np.linalg.eigh(cov_a)
    vals_b, vecs_b = np.linalg.eigh(cov_b)

    # 6) Halbachsen der Konfidenzellipse
    axes_a = np.sqrt(vals_a * chi_val)  # [a1, a2]
    axes_b = np.sqrt(vals_b * chi_val)  # [b1, b2]

    # 7) Vektor zwischen Zentren
    delta = mu_b - mu_a

    # 8) Projectionstest auf jede Hauptachse:
    #    Für jede Achse i prüfen, ob |delta·v_i| <= a_i + b_i
    for vec, ra, rb in zip(vecs_a.T, axes_a, axes_b):
        proj = abs(np.dot(delta, vec))
        if proj > (ra + rb):
            return False

    # Wenn auf allen Achsen kein Separationsabstand, dann überlappen
    return True



def compute_overlap_rhombus(row, p: float = 0.5) -> bool:
    """Return ``True`` if two rhombus confidence areas overlap.

    Parameters
    ----------
    row : pandas.Series
        Must contain ``coords_a_x``, ``coords_a_y``, ``coords_b_x`` and
        ``coords_b_y`` columns with list-like coordinate values (or string
        representations).
    p : float, default=0.5
        Probability mass used for the chi-square radius calculation.

    The function computes Manhattan (L1) distances of each point from its
    centroid, estimates a radius using ``chi2.ppf(p, df=2)`` and returns
    ``True`` if the distance between centroids does not exceed the sum of the
    two radii.
    """

    xa = parse_list(row["coords_a_x"])
    ya = parse_list(row["coords_a_y"])
    xb = parse_list(row["coords_b_x"])
    yb = parse_list(row["coords_b_y"])

    if len(xa) < 2 or len(xb) < 2 or len(xa) != len(ya) or len(xb) != len(yb):
        return False

    pts_a = np.column_stack([xa, ya])
    pts_b = np.column_stack([xb, yb])
    mu_a = pts_a.mean(axis=0)
    mu_b = pts_b.mean(axis=0)

    dist_a = np.abs(pts_a - mu_a).sum(axis=1)
    dist_b = np.abs(pts_b - mu_b).sum(axis=1)

    std_a = dist_a.std(ddof=1)
    std_b = dist_b.std(ddof=1)
    chi_val = chi2.ppf(p, df=2)
    r1 = chi_val * std_a
    r2 = chi_val * std_b
    center_dist = np.linalg.norm(mu_a - mu_b)
    return center_dist <= (r1 + r2)


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


def compute_bcr(cm: pd.DataFrame) -> float:
    """Return the Balanced Classification Rate (BCR) for ``cm``.

    BCR as described in the FIT documentation is the mean of the true
    positive rate and the true negative rate calculated from the confusion
    matrix produced by :func:`compute_confusion`.
    """

    tpr = cm.loc["true_same", "pred_same"] / cm.loc["true_same"].sum()
    tnr = cm.loc["true_diff", "pred_diff"] / cm.loc["true_diff"].sum()
    return (tpr + tnr) / 2


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
