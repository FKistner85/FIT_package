from __future__ import annotations

"""Population size estimation helpers."""

from typing import Sequence

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform


def cluster_population(dist_matrix: pd.DataFrame, cutoff: float) -> int:
    """Return the number of clusters given a distance threshold.

    Parameters
    ----------
    dist_matrix : pd.DataFrame
        Square pairwise distance matrix.
    cutoff : float
        Distance threshold passed to :func:`scipy.cluster.hierarchy.fcluster`.

    Returns
    -------
    int
        Detected cluster count.
    """
    condensed = squareform(dist_matrix.to_numpy(), checks=False)
    link = linkage(condensed, method="ward")
    labels = fcluster(link, t=cutoff, criterion="distance")
    return int(np.unique(labels).size)


def compute_erd(predicted: int, true: int) -> float:
    """Compute the expected relative difference (ERD).

    The ERD measures how far ``predicted`` deviates from ``true`` relative
    to the ground truth. It is defined as ``abs(predicted - true) / true``.

    Parameters
    ----------
    predicted : int
        Estimated population size.
    true : int
        Known population size.

    Returns
    -------
    float
        Relative deviation between prediction and truth.
    """
    if true <= 0:
        raise ValueError("true must be positive")
    return abs(predicted - true) / float(true)


def optimal_cutoff(distances: pd.DataFrame, true_n: int) -> tuple[float, tuple[float, float]]:
    """Return the Ward distance most likely to yield ``true_n`` clusters.

    The confidence interval describes the range of distances producing the
    same cluster count. Its bounds correspond to the 25\% and 75\%
    percentiles of that interval.

    Parameters
    ----------
    distances : pd.DataFrame
        Square pairwise distance matrix.
    true_n : int
        Expected number of clusters in the data.

    Returns
    -------
    tuple
        ``(cutoff, (low, high))`` where ``cutoff`` is the midpoint of the
        valid distance range and ``(low, high)`` is the 25\% confidence
        interval around it.
    """

    if true_n <= 0:
        raise ValueError("true_n must be positive")

    condensed = squareform(distances.to_numpy(), checks=False)
    link = linkage(condensed, method="ward")
    merge_dists = link[:, 2]

    n_obs = distances.shape[0]
    if true_n > n_obs:
        raise ValueError("true_n cannot exceed number of observations")

    lower_index = n_obs - true_n - 1
    lower = merge_dists[lower_index] if lower_index >= 0 else 0.0

    upper_index = n_obs - true_n
    upper = (
        merge_dists[upper_index]
        if upper_index < len(merge_dists)
        else float("inf")
    )

    width = upper - lower if np.isfinite(upper) else 0.0
    cutoff = lower + 0.5 * width
    ci = (lower + 0.25 * width, lower + 0.75 * width)
    return cutoff, ci


def concordance_correlation_coefficient(x: Sequence[float], y: Sequence[float]) -> float:
    """Return Lin's concordance correlation coefficient between ``x`` and ``y``."""

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size != y.size:
        raise ValueError("x and y must have the same length")

    mean_x = x.mean()
    mean_y = y.mean()
    var_x = x.var(ddof=1)
    var_y = y.var(ddof=1)
    cov_xy = np.cov(x, y, ddof=1)[0, 1]

    numerator = 2 * cov_xy
    denominator = var_x + var_y + (mean_x - mean_y) ** 2
    if denominator == 0:
        return np.nan
    return float(numerator / denominator)
