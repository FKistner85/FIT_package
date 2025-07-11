import numpy as np
from scipy.spatial.distance import (
    euclidean,
    cityblock,
    cosine,
    mahalanobis,
    chebyshev,
    canberra,
    braycurtis,
)


def compute_distances(a, b, VI=None):
    """Return several distance metrics between vectors ``a`` and ``b``.

    Parameters
    ----------
    a, b : array-like
        Input vectors.
    VI : array-like, optional
        Inverse covariance matrix for Mahalanobis distance.
    """
    distances = {
        "euclidean": euclidean(a, b),
        "manhattan": cityblock(a, b),
        "cosine": cosine(a, b),
        "chebyshev": chebyshev(a, b),
        "canberra": canberra(a, b),
        "braycurtis": braycurtis(a, b),
    }

    return distances
