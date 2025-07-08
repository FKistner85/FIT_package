# distance_metrics.py

## Overview
Provides `compute_distances` returning several metric distances.

## Key Components
- compute_distances

### compute_distances
Takes two numeric vectors and returns a dictionary with common distance metrics
such as Euclidean, Manhattan and Cosine. Pass the inverse covariance matrix via
`VI` to enable Mahalanobis distance. Use this helper when comparing projected
feature vectors. It assumes SciPy is installed and does not operate on entire
matrices so larger computations must loop over rows manually.

## References
- https://docs.scipy.org/doc/scipy/reference/spatial.distance.html

## Assumptions and Limitations
Mahalanobis requires an inverse covariance matrix.
