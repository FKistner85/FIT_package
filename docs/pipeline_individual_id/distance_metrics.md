# distance_metrics.py

`compute_distances()` calculates several metrics between two vectors. The implementation shows the supported functions:

```python
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
```
【F:src/FIT_python/pipeline_individual_id/distance_metrics.py†L12-L23】

* **Euclidean** and **Manhattan** are standard $ℓ_2$ and $ℓ_1$ distances.
* **Cosine** measures angular similarity and is scale invariant.
* **Chebyshev** focuses on the largest absolute difference.
* **Canberra** and **Bray‑Curtis** weight differences relative to absolute values.
* **Mahalanobis** accounts for covariance when a matrix $VI$ is provided.
