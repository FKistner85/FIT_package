# distance_metrics.py

## Overview
Provides `compute_distances` returning several metric distances.

## Key Components
- compute_distances

### compute_distances
Takes two numeric vectors and returns a dictionary with common distance metrics
such as Euclidean, Manhattan, Cosine, Chebyshev, Canberra and Bray‑Curtis.
Use this helper when comparing projected feature vectors. It assumes SciPy is
installed and does not operate on entire matrices so larger computations must
loop over rows manually.

## References
- https://docs.scipy.org/doc/scipy/reference/spatial.distance.html

## Assumptions and Limitations
The function returns the listed distances for individual vector pairs and does
not support processing entire matrices in one call.
