from __future__ import annotations

"""Utility helpers for the individual ID pipelines."""

from ast import literal_eval
from typing import Iterable, List, Dict
from FIT_python.soft_config import SOFT_CONFIG

import numpy as np
from scipy.stats import chi2


def parse_list(s: str) -> np.ndarray:
    """Safely parse a list representation to a NumPy array."""
    try:
        return np.asarray(literal_eval(s), dtype=float)
    except Exception:
        return np.empty(0, dtype=float)


def compute_overlap_jsl_style(row, p: float = 0.5) -> bool:
    """Return ``True`` if two ellipses overlap.

    The implementation mirrors the logic used in the original JSL tool. Each
    row must provide the columns ``coords_a_x``, ``coords_a_y``, ``coords_b_x``
    and ``coords_b_y`` containing string representations of coordinate lists.
    The ``p`` parameter controls the chi-square factor (default ``0.5``).
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

    dist_a = np.linalg.norm(pts_a - mu_a, axis=1)
    dist_b = np.linalg.norm(pts_b - mu_b, axis=1)

    std_a = dist_a.std(ddof=1)
    std_b = dist_b.std(ddof=1)
    chi_val = np.sqrt(chi2.ppf(p, df=2))
    r1 = chi_val * std_a
    r2 = chi_val * std_b
    center_dist = np.linalg.norm(mu_a - mu_b)
    return center_dist <= (r1 + r2)


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
