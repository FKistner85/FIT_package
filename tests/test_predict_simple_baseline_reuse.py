import pandas as pd
import importlib
import pytest


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
