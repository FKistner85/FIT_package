import pandas as pd
from FIT_python.pipeline_individual_id.population_estimation import (
    cluster_population,
    compute_erd,
)


def test_cluster_population_simple():
    data = [
        [0.0, 0.1, 1.0, 1.1],
        [0.1, 0.0, 1.0, 1.1],
        [1.0, 1.0, 0.0, 0.2],
        [1.1, 1.1, 0.2, 0.0],
    ]
    df = pd.DataFrame(data, index=range(4), columns=range(4))
    n_clusters = cluster_population(df, cutoff=0.5)
    assert n_clusters == 2


def test_compute_erd():
    assert compute_erd(5, 4) == 0.25
