import pandas as pd
from FIT_python.pipeline_individual_id.sequential_cv import sequential_cv_test_pairs

def build_data():
    rows = []
    for i in range(13):
        ind = chr(ord('A') + i)
        for j in range(3):
            rows.append({'individual_id': ind, 'sex': 0, 'trail': f'{ind}{j}', 'f1': float(i) + j*0.1, 'f2': float(i) + j*0.2})
    return pd.DataFrame(rows)


def test_sequential_cv_runs():
    df = build_data()
    summary = sequential_cv_test_pairs(
        df,
        ['f1', 'f2'],
        n_repeats=1,
        seed=0,
        reducer='pca',
        pair_kwargs={'window_lengths': [2], 'n_windows': 1, 'N_pool': 5, 'n_folds': 2}
    )
    assert isinstance(summary, pd.DataFrame)
