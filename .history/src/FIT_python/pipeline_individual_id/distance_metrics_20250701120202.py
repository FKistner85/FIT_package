import numpy as np
from scipy.spatial.distance import (
    euclidean, cityblock, cosine,
    chebyshev, canberra, braycurtis
)

def compute_distances(a, b):
    """
    Berechnet verschiedene Distanzen zwischen zwei Vektoren a und b.
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
