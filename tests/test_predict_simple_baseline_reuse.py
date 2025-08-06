import pandas as pd
import importlib
import numpy as np
import pytest
from pathlib import Path


def test_predict_simple_baseline_reuse_csv(tmp_path, monkeypatch):
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
    from FIT_python.utils import get_species_paths

    paths = get_species_paths('otter')
    csv_file = paths['predictions'] / 'otter_baseline_predictions.csv'
    csv_file.parent.mkdir(parents=True, exist_ok=True)
    data = pd.DataFrame({'a': [1], 'b': [2]})
    data.to_csv(csv_file, index=False)

    import FIT_python.pipeline_sex.simple_baseline as sb
    importlib.reload(sb)

    df = sb.predict_simple_baseline('otter', reuse_csv=True)
    pd.testing.assert_frame_equal(df, data)


def test_predict_simple_baseline_missing_csv(tmp_path, monkeypatch):
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
    import FIT_python.pipeline_sex.simple_baseline as sb
    importlib.reload(sb)

    from FIT_python.utils import get_species_paths
    paths = get_species_paths('otter')
    csv_file = paths['predictions'] / 'otter_baseline_predictions.csv'
    if csv_file.exists():
        csv_file.unlink()

    with pytest.raises(FileNotFoundError):
        sb.predict_simple_baseline('otter', reuse_csv=True)


def _setup_env(tmp_path, monkeypatch):
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
    from FIT_python.utils import get_species_paths

    paths = get_species_paths('otter')
    df = pd.DataFrame({'sex': ['f', 'm'], 'a': [0, 1]})
    paths['splits'].mkdir(parents=True, exist_ok=True)
    df.to_parquet(paths['splits'] / 'train.parquet')
    df.to_parquet(paths['splits'] / 'test.parquet')
    return root, paths


def test_predict_simple_baseline_uses_species_models(tmp_path, monkeypatch):
    root, paths = _setup_env(tmp_path, monkeypatch)

    import joblib
    called = {}

    class Dummy:
        def predict(self, X):
            return np.array(['f'] * len(X))

        def predict_proba(self, X):
            return np.tile([[1.0, 0.0]], (len(X), 1))

    def fake_load(path):
        called['path'] = Path(path)
        return Dummy()

    monkeypatch.setattr(joblib, 'load', fake_load)

    import FIT_python.pipeline_sex.simple_baseline as sb
    importlib.reload(sb)

    df = sb.predict_simple_baseline('otter', reuse_csv=False)
    assert called['path'] == paths['models'] / 'otter.joblib'
    csv_file = paths['predictions'] / 'otter_baseline_predictions.csv'
    assert csv_file.exists()
    pd.testing.assert_frame_equal(df, pd.read_csv(csv_file), check_dtype=False)


def test_predict_simple_baseline_uses_exp_dir_models(tmp_path, monkeypatch):
    root, paths = _setup_env(tmp_path, monkeypatch)

    import joblib
    called = {}

    class Dummy:
        def predict(self, X):
            return np.array(['f'] * len(X))

        def predict_proba(self, X):
            return np.tile([[1.0, 0.0]], (len(X), 1))

    def fake_load(path):
        called['path'] = Path(path)
        return Dummy()

    monkeypatch.setattr(joblib, 'load', fake_load)

    import FIT_python.pipeline_sex.simple_baseline as sb
    importlib.reload(sb)

    exp_dir = root / 'exp'
    df = sb.predict_simple_baseline('otter', exp_dir=exp_dir, reuse_csv=False)
    assert called['path'] == exp_dir / 'models' / 'otter.joblib'
    csv_file = paths['predictions'] / 'otter_baseline_predictions.csv'
    assert csv_file.exists()
    pd.testing.assert_frame_equal(df, pd.read_csv(csv_file), check_dtype=False)
