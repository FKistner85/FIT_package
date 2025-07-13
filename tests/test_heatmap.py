import pandas as pd
from pathlib import Path
from FIT_python.pipeline_sex.sex_predict_and_visualisation import plot_hyperparam_heatmap

def test_heatmap_accepts_bayessearch_columns(tmp_path):
    df = pd.DataFrame({
        'select__method': ['forward', 'lasso'],
        'reduce_pre__method': ['pca', 'umap'],
        'mean_test_score': [0.8, 0.75],
    })
    out = plot_hyperparam_heatmap(df, tmp_path)
    assert out.exists()


def test_heatmap_defaults_missing_columns(tmp_path):
    df = pd.DataFrame({
        "fs_method": ["forward", "lasso"],
        "cv_balanced_accuracy": [0.8, 0.75],
    })
    out = plot_hyperparam_heatmap(df, tmp_path)
    assert out.exists()
