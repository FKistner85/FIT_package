import numpy as np
from scipy.spatial.distance import (
    euclidean, cityblock, cosine,
    chebyshev, canberra, braycurtis
)

def compute_distances(a, b):
    """Return several distance metrics between vectors ``a`` and ``b``.

    Parameters
    ----------
    a, b : array-like
        Input vectors.
    """
    distances = {
        "euclidean": euclidean(a, b),
        "manhattan": cityblock(a, b),
        "cosine": cosine(a, b),
        "chebyshev": chebyshev(a, b),
        "canberra": canberra(a, b),
        "braycurtis": braycurtis(a, b),
    }
    
