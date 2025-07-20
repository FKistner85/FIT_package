import pandas as pd
import importlib


def test_predict_simple_baseline_reuse_csv(tmp_path, monkeypatch):
    root = tmp_path
    csv_dir = root / 'exp'
    csv_dir.mkdir(parents=True)
    (root / 'data').mkdir()
    csv_file = csv_dir / 'otter_baseline_predictions.csv'
    data = pd.DataFrame({'a': [1], 'b': [2]})
    data.to_csv(csv_file, index=False)

    monkeypatch.setenv('FIT_PROJECT_ROOT', str(root))
    import FIT_python.config as cfg
    importlib.reload(cfg)
    import FIT_python.pipeline_sex.simple_baseline as sb
    importlib.reload(sb)

    df = sb.predict_simple_baseline('otter', csv_dir, reuse_csv=True)
    pd.testing.assert_frame_equal(df, data)
