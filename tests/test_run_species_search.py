import pandas as pd
import importlib
import types
import sys


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


def test_run_species_search_executes(tmp_path, monkeypatch):
    root = tmp_path
    data_dir = root / "data" / "splits" / "otter"
    data_dir.mkdir(parents=True)
    build_train_df().to_parquet(data_dir / "train.parquet", index=False)
    build_test_df().to_parquet(data_dir / "test.parquet", index=False)

    monkeypatch.setenv("FIT_EXPERIMENT_ROOT", str(root))
    import FIT_python.config as cfg
    importlib.reload(cfg)
    import FIT_python.soft_config as scfg
    scfg.SOFT_CONFIG["pipeline_sex"]["run_otter_search_sex"]["cv"] = "fold"
    import FIT_python.pipeline_sex.sex_config as sc
    importlib.reload(sc)
    captured = {}

    def fake_run_species_search(species, base_dir_suffix, n_iter, cv, random_state):
        species_dir = sc.SPLITS_DIR / species
        df_train = pd.read_parquet(species_dir / "train.parquet").query("sex in ['f','m']")
        if cv == "fold":
            ps = sc.PredefinedSplit(test_fold=df_train["Fold"].astype(int).to_numpy())
            captured["cv"] = ps
        else:
            captured["cv"] = cv

    monkeypatch.setattr(sc, "_run_species_search", fake_run_species_search)

    sc.run_species_search()
    assert isinstance(captured["cv"], sc.PredefinedSplit)
