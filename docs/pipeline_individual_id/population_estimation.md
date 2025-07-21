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

`optimal_cutoff()` locates the Ward distance that most confidently
yields a target number of clusters and reports a 25% confidence interval:

```python
    def optimal_cutoff(distances: pd.DataFrame, true_n: int) -> tuple[float, tuple[float, float]]:
        """Return the Ward distance most likely to yield ``true_n`` clusters.
        ...
        ``(cutoff, (low, high))`` is returned where the interval corresponds to
        the 25% and 75% percentiles of valid distances.
        """
```
【F:src/FIT_python/pipeline_individual_id/population_estimation.py†L56-L103】

`concordance_correlation_coefficient()` implements Lin's coefficient to compare
two measurements:

```python
    def concordance_correlation_coefficient(x: Sequence[float], y: Sequence[float]) -> float:
        """Return Lin's concordance correlation coefficient between ``x`` and ``y``."""
```
【F:src/FIT_python/pipeline_individual_id/population_estimation.py†L106-L124】

`silhouette_cluster_count()` scans a range of ``k`` values and selects the
number of clusters that maximises the silhouette score:

```python
    def silhouette_cluster_count(dist_matrix: pd.DataFrame) -> int:
        """Return the Ward cluster count with the highest silhouette score."""
```
【F:src/FIT_python/pipeline_individual_id/population_estimation.py†L127-L162】
