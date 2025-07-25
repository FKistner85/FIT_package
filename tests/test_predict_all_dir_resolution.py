import pandas as pd
import importlib


def _setup(tmp_path, monkeypatch):
    root = tmp_path
    models_dir = root / 'results' / 'data' / 'random_search_standard_metrics'
    models_dir.mkdir(parents=True)
    (root / 'data').mkdir()
    csv_file = models_dir / 'otter_all_predictions.csv'
    pd.DataFrame({'a': [1]}).to_csv(csv_file, index=False)

    monkeypatch.setenv('FIT_EXPERIMENT_ROOT', str(root))
    monkeypatch.setenv('FIT_RAW_DIR', str(root / 'data' / 'raw'))
    import FIT_python.config as cfg
    importlib.reload(cfg)
    import FIT_python.pipeline_sex.sex_predict_and_visualisation as sp
    importlib.reload(sp)
    return sp


def test_predict_all_resolves_experiment_root(tmp_path, monkeypatch):
    sp = _setup(tmp_path, monkeypatch)
    df = sp.predict_all('otter', models_dir=tmp_path, reuse_csv=True)
    pd.testing.assert_frame_equal(df, pd.DataFrame({'a': [1]}))


def test_predict_all_resolves_models_subdir(tmp_path, monkeypatch):
    sp = _setup(tmp_path, monkeypatch)
    df = sp.predict_all('otter', models_dir=tmp_path / 'models', reuse_csv=True)
    pd.testing.assert_frame_equal(df, pd.DataFrame({'a': [1]}))
