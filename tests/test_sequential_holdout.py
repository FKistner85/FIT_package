import pandas as pd
from FIT_python.pipeline_individual_id.sequential_holdout import run


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
