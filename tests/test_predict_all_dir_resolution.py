import pandas as pd
import importlib
from FIT_python.utils import get_species_paths


def _setup(tmp_path, monkeypatch):
    root = tmp_path
    (root / 'data').mkdir()
    monkeypatch.setenv('FIT_EXPERIMENT_ROOT', str(root))
    monkeypatch.setenv('FIT_RAW_DIR', str(root / 'data' / 'raw'))
    import FIT_python.config as cfg
    importlib.reload(cfg)
    import FIT_python.utils.paths as paths_mod
    importlib.reload(paths_mod)
    import FIT_python.utils as utils
    importlib.reload(utils)

    paths = get_species_paths('otter')
    csv_file = paths['predictions'] / 'otter_all_predictions.csv'
    csv_file.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({'a': [1]}).to_csv(csv_file, index=False)

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
