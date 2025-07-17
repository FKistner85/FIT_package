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
