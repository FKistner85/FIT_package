import pandas as pd
import importlib
import pytest

def test_predict_all_reuse_csv(tmp_path, monkeypatch):
    root = tmp_path
    csv_dir = root / 'results' / 'data' / 'random_search_standard_metrics'
    csv_dir.mkdir(parents=True)
    (root / 'data').mkdir()
    csv_file = csv_dir / 'otter_all_predictions.csv'
    data = pd.DataFrame({'a':[1], 'b':[2]})
    data.to_csv(csv_file, index=False)

    monkeypatch.setenv('FIT_PROJECT_ROOT', str(root))
    import FIT_python.config as cfg
    importlib.reload(cfg)
    import FIT_python.pipeline_sex.sex_predict_and_visualisation as sp
    importlib.reload(sp)

    df = sp.predict_all('otter', reuse_csv=True)
    pd.testing.assert_frame_equal(df, data)


def test_predict_all_missing_csv(tmp_path, monkeypatch):
    root = tmp_path
    (root / 'results' / 'data' / 'random_search_standard_metrics').mkdir(parents=True)
    (root / 'data').mkdir()

    monkeypatch.setenv('FIT_PROJECT_ROOT', str(root))
    import FIT_python.config as cfg
    importlib.reload(cfg)
    import FIT_python.pipeline_sex.sex_predict_and_visualisation as sp
    importlib.reload(sp)

    with pytest.raises(FileNotFoundError):
        sp.predict_all('otter', reuse_csv=True)
