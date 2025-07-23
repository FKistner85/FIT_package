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
    import FIT_python.pipeline_sex.sex_config as sc
    importlib.reload(sc)
    # run sequentially to avoid import issues in subprocesses
    from skopt import BayesSearchCV as _BSCV
    def _wrapper(*a, **kw):
        kw["n_jobs"] = 1
        return _BSCV(*a, **kw)
    sc.BayesSearchCV = _wrapper

    sc.run_species_search(n_iter=1, cv=2, random_state=0)
