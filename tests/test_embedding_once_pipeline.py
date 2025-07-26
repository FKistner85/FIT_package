import pandas as pd
import pytest
from FIT_python.pipeline_individual_id.pairwise_individual_id_pipeline import (
    run_all_pairwise_projections_parallel,
    run_embedding_once_pipeline,
)

def build_data():
    rows = []
    rows += [{'individual_id': 'A', 'sex': 0, 'f1': 0.0, 'f2': 0.0} for _ in range(2)]  # A1
    rows += [{'individual_id': 'A', 'sex': 0, 'f1': 0.0, 'f2': 0.0} for _ in range(2)]  # A2
    rows += [{'individual_id': 'B', 'sex': 1, 'f1': 1.0, 'f2': 1.0} for _ in range(2)]  # B1
    df = pd.DataFrame(rows)
    df.insert(0, "id", list(range(len(df))))
    return df

def build_comparisons():
    return [
        {
            'ind_a': 'A',
            'ind_b': 'A',
            'trail_a_id': 'A2',
            'trail_b_id': 'A1',
            'samples_a': [2, 3],
            'samples_b': [0, 1],
            'same_individual': True,
            'fold': 0,
        },
        {
            'ind_a': 'A',
            'ind_b': 'B',
            'trail_a_id': 'A2',
            'trail_b_id': 'B1',
            'samples_a': [2, 3],
            'samples_b': [4, 5],
            'same_individual': False,
            'fold': 0,
        },
    ]

def test_embedding_once_matches_pairwise():
    df = build_data()
    comps = build_comparisons()

    res_pw = run_all_pairwise_projections_parallel(
        comps,
        df,
        feature_cols=['f1', 'f2'],
        k_features=2,
        reducers=['pca'],
        selection_method='forward',
        n_components=2,
        n_jobs=1,
    )

    res_emb = run_embedding_once_pipeline(
        comps,
        df,
        feature_cols=['f1', 'f2'],
        k_features=2,
        reducer='pca',
        selection_method='forward',
        n_components=2,
    )

    dist_pw = {(r['trail_a_id'], r['trail_b_id']): r['dist_euclidean'] for r in res_pw}
    dist_emb = {(r['trail_a_id'], r['trail_b_id']): r['dist_euclidean'] for r in res_emb}
    assert dist_pw == dist_emb


def test_resume_checkpoint(tmp_path):
    df = build_data()
    comps = build_comparisons()

    full = run_all_pairwise_projections_parallel(
        comps,
        df,
        feature_cols=['f1', 'f2'],
        k_features=2,
        reducers=['pca'],
        selection_method='forward',
        n_components=2,
        n_jobs=1,
    )

    ckpt = tmp_path / "state.joblib"
    run_all_pairwise_projections_parallel(
        comps[:1],
        df,
        feature_cols=['f1', 'f2'],
        k_features=2,
        reducers=['pca'],
        selection_method='forward',
        n_components=2,
        n_jobs=1,
        checkpoint_path=str(ckpt),
    )

    resumed = run_all_pairwise_projections_parallel(
        comps,
        df,
        feature_cols=['f1', 'f2'],
        k_features=2,
        reducers=['pca'],
        selection_method='forward',
        n_components=2,
        n_jobs=1,
        checkpoint_path=str(ckpt),
        resume=True,
    )

    assert resumed == full


def test_str_reducer_handling():
    df = build_data()
    comps = build_comparisons()

    res = run_all_pairwise_projections_parallel(
        comps,
        df,
        feature_cols=["f1", "f2"],
        k_features=2,
        reducers="pca",
        selection_method="forward",
        n_components=2,
        n_jobs=1,
    )

    assert len(res) == len(comps)


def test_n_components_none_raises():
    df = build_data()
    comps = build_comparisons()

    with pytest.raises(
        ValueError,
        match="n_components must be an int or list of ints, got None",
    ):
        run_all_pairwise_projections_parallel(
            comps,
            df,
            feature_cols=["f1", "f2"],
            k_features=2,
            reducers=["pca"],
            selection_method="forward",
            n_components=None,
            n_jobs=1,
        )
