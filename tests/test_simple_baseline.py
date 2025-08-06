import pandas as pd
import importlib


def build_train_df():
    return pd.DataFrame({
        "f1": [0.0, 1.0, 0.1, 1.1],
        "f2": [0.0, 1.0, 0.1, 1.1],
        "sex": ["f", "m", "f", "m"],
        "Fold": [0, 1, 0, 1],
        "individual_id": ["A", "B", "A2", "B2"],
    })


def build_test_df():
    return pd.DataFrame({
        "f1": [0.2, 1.2],
        "f2": [0.2, 1.2],
        "sex": ["f", "m"],
        "individual_id": ["A3", "B3"],
    })


def test_simple_baseline_runs(tmp_path, monkeypatch):
    root = tmp_path
    data_dir = root / "data" / "splits" / "otter"
    data_dir.mkdir(parents=True)
    build_train_df().to_parquet(data_dir / "train.parquet", index=False)
    build_test_df().to_parquet(data_dir / "test.parquet", index=False)

    monkeypatch.setenv("FIT_EXPERIMENT_ROOT", str(root))
    monkeypatch.setenv("FIT_RAW_DIR", str(root / "data" / "raw"))
    import FIT_python.config as cfg
    importlib.reload(cfg)
    import FIT_python.utils.paths as paths_mod
    importlib.reload(paths_mod)
    import FIT_python.utils as utils
    importlib.reload(utils)
    import FIT_python.pipeline_sex.simple_baseline as sb
    importlib.reload(sb)

    out_dir = root / "exp"
    df = sb.run_simple_baseline_all_species(out_dir, n_jobs=1)

    import FIT_python.utils.paths as paths_mod
    importlib.reload(paths_mod)
    import FIT_python.utils as utils
    importlib.reload(utils)
    from FIT_python.utils import get_species_paths
    paths = get_species_paths("otter")

    assert (out_dir / "raw_results.csv").exists()
    assert (out_dir / "models" / "otter.joblib").exists()
    pred_csv = paths["predictions"] / "otter_baseline_predictions.csv"
    assert pred_csv.exists()
    pd.read_csv(pred_csv)
    expected_cols = {
        "species",
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall",
        "f1",
        "female_individual_acc",
        "male_individual_acc",
        "balanced_individual_acc",
        "maj_correct",
        "maj_wrong",
        "maj_pct",
    }
    assert expected_cols.issubset(df.columns)


def test_simple_baseline_reuse_results(tmp_path, monkeypatch):
    root = tmp_path
    exp_dir = root / "exp"
    model_dir = exp_dir / "models"
    model_dir.mkdir(parents=True)
    data = pd.DataFrame({"species": ["otter"], "accuracy": [1.0]})
    data.to_csv(exp_dir / "raw_results.csv", index=False)
    (model_dir / "otter.joblib").write_bytes(b"0")
    (root / "data").mkdir()

    monkeypatch.setenv("FIT_EXPERIMENT_ROOT", str(root))
    monkeypatch.setenv("FIT_RAW_DIR", str(root / "data" / "raw"))
    import FIT_python.config as cfg
    importlib.reload(cfg)
    import FIT_python.pipeline_sex.simple_baseline as sb
    importlib.reload(sb)

    df = sb.run_simple_baseline_all_species(exp_dir, n_jobs=1, reuse_results=True)
    pd.testing.assert_frame_equal(df, data)
