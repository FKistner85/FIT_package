import numpy as np
from scipy.spatial.distance import (
    euclidean, cityblock, cosine, mahalanobis,
    chebyshev, canberra, braycurtis
)

def compute_distances(a, b, VI=None):
    """
    Berechnet verschiedene Distanzen zwischen zwei Vektoren a und b.
    Optional: Mahalanobis-Inverse-Covariance-Matrix (VI) übergeben.
    """
    distances = {
        "euclidean": euclidean(a, b),
        "manhattan": cityblock(a, b),
        "cosine": cosine(a, b),
        "chebyshev": chebyshev(a, b),
        "canberra": canberra(a, b),
        "braycurtis": braycurtis(a, b),
    }
    if VI is not None:
        distances["mahalanobis"] = mahalanobis(a, b, VI)
    else:
        distances["mahalanobis"] = np.nan
    return distances
