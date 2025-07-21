import pandas as pd
import pytest

from FIT_python.pipeline_individual_id.population_estimation import (
    cluster_population,
    compute_erd,
    optimal_cutoff,
    concordance_correlation_coefficient,
    silhouette_cluster_count,
)


def _build_matrix():
    data = [
        [0.0, 0.1, 1.0, 1.1],
        [0.1, 0.0, 1.0, 1.1],
        [1.0, 1.0, 0.0, 0.2],
        [1.1, 1.1, 0.2, 0.0],
    ]
    return pd.DataFrame(data, index=range(4), columns=range(4))


def test_cluster_population_simple():
    df = _build_matrix()
    n_clusters = cluster_population(df, cutoff=0.5)
    assert n_clusters == 2


def test_compute_erd():
    assert compute_erd(5, 4) == 0.25


def test_optimal_cutoff_simple():
    df = _build_matrix()
    cutoff, ci = optimal_cutoff(df, 2)
    assert pytest.approx(cutoff, rel=1e-6) == 0.839087275
    low, high = ci
    assert pytest.approx(low, rel=1e-6) == 0.519543638
    assert pytest.approx(high, rel=1e-6) == 1.158630912


def test_concordance_correlation_coefficient():
    assert concordance_correlation_coefficient([1, 2, 3], [1, 2, 3]) == 1.0
    assert concordance_correlation_coefficient([1, 1, 1], [2, 2, 2]) == 0.0


def test_silhouette_cluster_count():
    df = _build_matrix()
    assert silhouette_cluster_count(df) == 2
