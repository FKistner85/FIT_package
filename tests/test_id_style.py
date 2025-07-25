import pandas as pd
import importlib


def test_id_colors_populated(tmp_path, monkeypatch):
    root = tmp_path
    csv_dir = root / "experiments" / "fit_start_to_finish"
    csv_dir.mkdir(parents=True)
    (root / "data").mkdir()
    df = pd.DataFrame({"individual_id": ["A", "B", "A"], "trail": ["t1", "t2", "t3"]})
    df.to_csv(csv_dir / "otter_baseline_predictions.csv", index=False)

    monkeypatch.setenv("FIT_EXPERIMENT_ROOT", str(root))
    monkeypatch.setenv("FIT_RAW_DIR", str(root / "data" / "raw"))
    import FIT_python.config as cfg
    importlib.reload(cfg)
    import FIT_python.Visualisations.id_style as ids
    importlib.reload(ids)

    assert ids.ID_COLORS
    assert ids.TRAIL_TO_ID == {"t1": "a", "t2": "b", "t3": "a"}
