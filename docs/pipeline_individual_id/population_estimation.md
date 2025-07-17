# population_estimation.py

`cluster_population()` groups individuals based on a distance matrix and counts the resulting clusters:

```python
    def cluster_population(dist_matrix: pd.DataFrame, cutoff: float) -> int:
        """Return the number of clusters given a distance threshold.
        ...
        Distance threshold passed to :func:`scipy.cluster.hierarchy.fcluster`.
        ...
        Detected cluster count.
        """
```
【F:src/FIT_python/pipeline_individual_id/population_estimation.py†L13-L31】

`compute_erd()` compares the predicted population size to the known number of individuals:

```python
    def compute_erd(predicted: int, true: int) -> float:
        """Compute the expected relative difference (ERD)."""
```
【F:src/FIT_python/pipeline_individual_id/population_estimation.py†L34-L54】

Both functions can be combined to estimate a population from pairwise distances and report the deviation from ground truth:

```python
import pandas as pd

# distance matrix for N animals
dist_df = pd.DataFrame([...])

predicted_n = cluster_population(dist_df, cutoff=0.5)
true_n = 4
erd = compute_erd(predicted_n, true_n)
```

The ERD returns ``abs(predicted - true) / true`` so smaller values indicate a more accurate estimate.
