import pandas as pd
from FIT_python.pipeline_individual_id.sequential_holdout import run
from FIT_python.pipeline_individual_id.holdout_helper import generate_holdout_sets


def build_df():
    rows = []
    for i, ind in enumerate(['A', 'B', 'C']):
        for j in range(2):
            rows.append({
                'id': len(rows),
                'individual_id': ind,
                'f1': float(i),
                'f2': float(j),
                'trail': f"{ind}_{j}",
                'sex': 'F' if i % 2 == 0 else 'M',
                'fold': 0,
            })
    return pd.DataFrame(rows)


def test_run_produces_summary(tmp_path):
    df = build_df()
    preds = pd.DataFrame({'p_f': 0.5, 'p_m': 0.5}, index=df['id'])
    from FIT_python.soft_config import SOFT_CONFIG
    SOFT_CONFIG["pipeline_individual_id"]["trail_generation_defaults"]["sample_size"] = 1
    summary = run(df, ['f1', 'f2'], preds, val_sizes=[2], iterations=1, out_dir=tmp_path, n_jobs=1)
    assert not summary.empty
    assert 'bcr' in summary.columns
    assert 'erd' in summary.columns
    assert 'pred_count' in summary.columns
    assert 'true_count' in summary.columns
    assert 'ccc' in summary.columns
    assert (tmp_path / 'summary.csv').exists()


def test_run_passes_k_features(monkeypatch, tmp_path):
    df = build_df()
    preds = pd.DataFrame({'p_f': 0.5, 'p_m': 0.5}, index=df['id'])

    recorded = {}

    def fake_generate(df_val, **kwargs):
        return [
            {
                'ind_a': 'A',
                'ind_b': 'B',
                'trail_a_id': 'A_0',
                'trail_b_id': 'B_0',
                'samples_a': [0],
                'samples_b': [1],
                'same_individual': False,
                'fold': 0,
            }
        ], pd.DataFrame({'n_pairs': [1], 'n_trails': [2]})

    def fake_run_all(comps, base_df, feature_cols, **kwargs):
        recorded['k'] = kwargs.get('k_features')
        return [
            {
                'trail_a_id': 'A_0',
                'trail_b_id': 'B_0',
                'coords_a_x': '[0,1]',
                'coords_a_y': '[0,0]',
                'coords_b_x': '[0,1]',
                'coords_b_y': '[1,1]',
                'same_individual': False,
                'dist_euclidean': 1.0,
            }
        ]

    monkeypatch.setattr(
        'FIT_python.pipeline_individual_id.sequential_holdout.generate_pairwise_comparisons_from_df',
        fake_generate,
    )
    monkeypatch.setattr(
        'FIT_python.pipeline_individual_id.sequential_holdout.run_all_pairwise_projections_parallel',
        fake_run_all,
    )

    summary = run(
        df,
        ['f1', 'f2'],
        preds,
        val_sizes=[2],
        iterations=1,
        out_dir=tmp_path,
        n_jobs=1,
        k_features=7,
    )

    assert recorded.get('k') == 7
    assert not summary.empty


def test_generate_holdout_sets(tmp_path):
    df = build_df()
    species_dir = tmp_path / "otter"
    species_dir.mkdir()
    df.to_parquet(species_dir / "train.parquet", index=False)

    splits = generate_holdout_sets(tmp_path, val_sizes=[2], iterations=2, seed=0)
    assert "otter" in splits
    assert len(splits["otter"]) == 2
    for split in splits["otter"]:
        train_ids = set(split["train_df"]["individual_id"].unique())
        val_ids = set(split["val_df"]["individual_id"].unique())
        assert train_ids.isdisjoint(val_ids)
