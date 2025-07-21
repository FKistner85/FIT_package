import pandas as pd
import numpy as np
import pytest

from FIT_python.pipeline_individual_id.sequential_holdout import compute_global_cutoffs
from FIT_python.pipeline_individual_id.population_estimation import optimal_cutoff


def _build_matrix(scale: float = 1.0) -> pd.DataFrame:
    data = [
        [0.0, 0.1 * scale, 1.0 * scale, 1.1 * scale],
        [0.1 * scale, 0.0, 1.0 * scale, 1.1 * scale],
        [1.0 * scale, 1.0 * scale, 0.0, 0.2 * scale],
        [1.1 * scale, 1.1 * scale, 0.2 * scale, 0.0],
    ]
    return pd.DataFrame(data, index=range(4), columns=range(4))


def _pairs_from_matrix(mat: pd.DataFrame, split: int) -> pd.DataFrame:
    rows = []
    for i in range(len(mat)):
        for j in range(i + 1, len(mat)):
            rows.append({
                "trail_a_id": i,
                "trail_b_id": j,
                "dist_euclidean": mat.iloc[i, j],
                "split": split,
                "n_val": 2,
            })
    return pd.DataFrame(rows)


def test_compute_global_cutoffs(tmp_path):
    mat1 = _build_matrix()
    mat2 = _build_matrix(scale=2.0)

    df_all = pd.concat([
        _pairs_from_matrix(mat1, 0),
        _pairs_from_matrix(mat2, 1),
    ], ignore_index=True)
    csv_fp = tmp_path / "all_splits.csv"
    df_all.to_csv(csv_fp, index=False)

    result = compute_global_cutoffs(csv_fp)

    c1, ci1 = optimal_cutoff(mat1, 2)
    c2, ci2 = optimal_cutoff(mat2, 2)

    expected_mean = np.mean([c1, c2])
    expected_median = np.median([c1, c2])
    expected_low = np.mean([ci1[0], ci2[0]])
    expected_high = np.mean([ci1[1], ci2[1]])

    assert pytest.approx(result["cutoff_mean"], rel=1e-6) == expected_mean
    assert pytest.approx(result["cutoff_median"], rel=1e-6) == expected_median
    assert pytest.approx(result["cutoff_low"], rel=1e-6) == expected_low
    assert pytest.approx(result["cutoff_high"], rel=1e-6) == expected_high
