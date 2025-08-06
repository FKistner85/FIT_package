import pandas as pd
import importlib
import types
import sys
from pathlib import Path

from FIT_python.config import GLOBAL_RANDOM_SEED


def _stub_module(name, cls_name):
    mod = types.ModuleType(name)
    class Dummy:
        def __init__(self, *a, **k):
            pass
    setattr(mod, cls_name, Dummy)
    return mod

sys.modules.setdefault('xgboost', _stub_module('xgboost', 'XGBClassifier'))
sys.modules.setdefault('lightgbm', _stub_module('lightgbm', 'LGBMClassifier'))
sys.modules.setdefault('catboost', _stub_module('catboost', 'CatBoostClassifier'))


def build_train_df():
    return pd.DataFrame({
        "v1": [0.0, 1.0, 0.1, 1.1],
        "v2": [0.0, 1.0, 0.1, 1.1],
        "sex": ["f", "m", "f", "m"],
        "Fold": [0, 1, 0, 1],
        "individual_id": ["A", "B", "A2", "B2"],
    })


def build_test_df():
    return pd.DataFrame({
        "v1": [0.2, 1.2],
        "v2": [0.2, 1.2],
        "sex": ["f", "m"],
        "individual_id": ["A3", "B3"],
    })


def test_run_species_search_creates_predictions(tmp_path, monkeypatch):
    root = tmp_path
    data_dir = root / "data" / "splits" / "otter"
    data_dir.mkdir(parents=True)
    build_train_df().to_parquet(data_dir / "train.parquet", index=False)
    build_test_df().to_parquet(data_dir / "test.parquet", index=False)

    monkeypatch.setenv("FIT_EXPERIMENT_ROOT", str(root))
    monkeypatch.setenv("FIT_RAW_DIR", str(root / "data" / "raw"))
    import FIT_python.config as cfg
    importlib.reload(cfg)
    import FIT_python.pipeline_sex.sex_config as sc
    importlib.reload(sc)

    class DummySearch:
        def __init__(self, estimator=None, search_spaces=None, n_iter=1, scoring=None, refit=None, cv=None, n_jobs=None, random_state=None, verbose=None):
            self.cv = cv
            self.n_iter = n_iter
            self.cv_results_ = {
                "params": [{}],
                "mean_test_accuracy": [1.0],
                "mean_test_balanced_accuracy": [1.0],
                "mean_test_neg_log_loss": [0.0],
            }

        def fit(self, X, y):
            return self

    monkeypatch.setattr(sc, "BayesSearchCV", DummySearch)
    monkeypatch.setattr(sc, "plot_hyperparam_heatmap", lambda *a, **k: None)

    def fake_dump(obj, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_bytes(b"0")
    monkeypatch.setattr(sc.joblib, "dump", fake_dump)

    def fake_predict_all(
        species,
        models_dir,
        reuse_csv=False,
        prefer_generic=False,
        include_inference=False,
        use_cv_train_predictions=False,
        metric_key=None,
    ):
        csv = Path(models_dir) / f"{species}_all_predictions.csv"
        if reuse_csv:
            return pd.read_csv(csv)
        csv.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame({"a": [1]})
        df.to_csv(csv, index=False)
        return df

    monkeypatch.setattr(sc, "predict_all", fake_predict_all)

    df_all, df_best = sc._run_species_search(
        "otter", "otter_bayes_search_standard_metrics", n_iter=1, cv=2, random_state=GLOBAL_RANDOM_SEED, reuse_results=False
    )
    assert isinstance(df_all, pd.DataFrame)
    assert isinstance(df_best, pd.DataFrame)

    from FIT_python.utils import get_species_paths
    csv_path = get_species_paths("otter")["search"] / "otter_all_predictions.csv"
    assert csv_path.exists()

    df = sc.predict_all("otter", models_dir=get_species_paths("otter")["search"], reuse_csv=True)
    pd.testing.assert_frame_equal(df, pd.DataFrame({"a": [1]}))


def test_run_species_search_copies_best_model(tmp_path, monkeypatch):
    root = tmp_path
    data_dir = root / "data" / "splits" / "otter"
    data_dir.mkdir(parents=True)
    build_train_df().to_parquet(data_dir / "train.parquet", index=False)
    build_test_df().to_parquet(data_dir / "test.parquet", index=False)

    monkeypatch.setenv("FIT_EXPERIMENT_ROOT", str(root))
    monkeypatch.setenv("FIT_RAW_DIR", str(root / "data" / "raw"))
    import FIT_python.config as cfg
    importlib.reload(cfg)
    import FIT_python.pipeline_sex.sex_config as sc
    importlib.reload(sc)

    class DummySearch:
        def __init__(self, estimator=None, search_spaces=None, n_iter=1, scoring=None, refit=None, cv=None, n_jobs=None, random_state=None, verbose=None):
            self.cv = cv
            self.n_iter = n_iter
            self.cv_results_ = {
                "params": [{}],
                "mean_test_accuracy": [1.0],
                "mean_test_balanced_accuracy": [1.0],
                "mean_test_neg_log_loss": [0.0],
            }

        def fit(self, X, y):
            return self

    monkeypatch.setattr(sc, "BayesSearchCV", DummySearch)
    monkeypatch.setattr(sc, "plot_hyperparam_heatmap", lambda *a, **k: None)

    def fake_dump(obj, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_bytes(b"0")
    monkeypatch.setattr(sc.joblib, "dump", fake_dump)

    def fake_predict_all(
        species,
        models_dir,
        reuse_csv=False,
        prefer_generic=False,
        include_inference=False,
        use_cv_train_predictions=False,
        metric_key=None,
    ):
        csv = Path(models_dir) / f"{species}_all_predictions.csv"
        if reuse_csv:
            return pd.read_csv(csv)
        csv.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame({"a": [1]})
        df.to_csv(csv, index=False)
        return df

    monkeypatch.setattr(sc, "predict_all", fake_predict_all)

    df_all, df_best = sc._run_species_search(
        "otter", "otter_bayes_search_standard_metrics", n_iter=1, cv=2, random_state=GLOBAL_RANDOM_SEED, reuse_results=False
    )
    assert isinstance(df_all, pd.DataFrame)
    assert isinstance(df_best, pd.DataFrame)

    from FIT_python.utils import get_species_paths
    model_path = get_species_paths("otter")["search"] / "best_mean_rank" / "otter.joblib"
    assert model_path.exists()


def test_run_species_search_reuses_results(tmp_path, monkeypatch):
    root = tmp_path
    data_dir = root / "data" / "splits" / "otter"
    data_dir.mkdir(parents=True)
    build_train_df().to_parquet(data_dir / "train.parquet", index=False)
    build_test_df().to_parquet(data_dir / "test.parquet", index=False)

    monkeypatch.setenv("FIT_EXPERIMENT_ROOT", str(root))
    monkeypatch.setenv("FIT_RAW_DIR", str(root / "data" / "raw"))
    import FIT_python.config as cfg
    importlib.reload(cfg)
    results_dir = cfg.RESULTS_DATA_DIR / "otter_bayes_search_standard_metrics"
    results_dir.mkdir(parents=True, exist_ok=True)
    df_all = pd.DataFrame({"a": [1]})
    df_best = pd.DataFrame({"b": [2]})
    df_all.to_csv(results_dir / "all_results.csv", index=False)
    df_best.to_csv(results_dir / "best_models.csv", index=False)
    import FIT_python.pipeline_sex.sex_config as sc
    importlib.reload(sc)

    class DummySearch:
        def __init__(self, *a, **k):
            pass

        def fit(self, X, y):
            raise RuntimeError("should not fit")

    monkeypatch.setattr(sc, "BayesSearchCV", DummySearch)
    monkeypatch.setattr(sc, "plot_hyperparam_heatmap", lambda *a, **k: None)

    got_all, got_best = sc._run_species_search(
        "otter",
        "otter_bayes_search_standard_metrics",
        n_iter=1,
        cv=2,
        random_state=GLOBAL_RANDOM_SEED,
        reuse_results=True,
    )

    pd.testing.assert_frame_equal(got_all, df_all)
    pd.testing.assert_frame_equal(got_best, df_best)
