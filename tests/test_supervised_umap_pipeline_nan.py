import pandas as pd
import numpy as np
import FIT_python.pipeline_individual_id.supervised_umap_pipeline as sup

class DummyPrep:
    def transform(self, X):
        return X

def dummy_prepare(df, feature_cols, sex_features, **kwargs):
    X = df[feature_cols].reset_index(drop=True)
    return X, DummyPrep()

class DummyReducer:
    def fit_transform(self, X, y=None):
        self._cols = [f"r{i}" for i in range(X.shape[1])]
        return X.to_numpy()
    def transform(self, X):
        return X.to_numpy()
    def get_feature_names_out(self):
        return self._cols

def dummy_compute(emb, comps):
    return pd.DataFrame([
        {"trail_a_id": "a1", "trail_b_id": "b1", "same_individual": True,
         "dist_euclidean": 0.5, "mean_euclidean_between": 0.5,
         "median_euclidean_between": 0.5},
        {"trail_a_id": "a2", "trail_b_id": "b2", "same_individual": False,
         "dist_euclidean": 1.0, "mean_euclidean_between": 1.0,
         "median_euclidean_between": 1.0},
        {"trail_a_id": "a3", "trail_b_id": "b3", "same_individual": True,
         "dist_euclidean": np.nan, "mean_euclidean_between": np.nan,
         "median_euclidean_between": np.nan},
    ])

class DummyLogReg:
    def fit(self, X, y):
        return self
    def predict_proba(self, X):
        return np.tile([0.2, 0.8], (len(X), 1))

def test_run_drops_nan(monkeypatch):
    monkeypatch.setattr(sup, "_prepare_features", dummy_prepare)
    monkeypatch.setattr(sup, "DimensionalityReducerTransformer", lambda *a, **k: DummyReducer())
    monkeypatch.setattr(sup, "_compute_pair_features", dummy_compute)
    monkeypatch.setattr(sup, "LogisticRegression", lambda *a, **k: DummyLogReg())

    train_df = pd.DataFrame({"id": [0, 1], "individual_id": ["A", "B"],
                             "f1": [0.0, 1.0], "f2": [0.0, 1.0]})
    comps = [{"trail_a_id": "ta", "trail_b_id": "tb", "samples_a": [0], "samples_b": [1], "same_individual": False}]

    res, _ = sup.run(train_df, comps, ["f1", "f2"])
    assert len(res) == 2
    assert not res[[c for c in res.columns if c.startswith("dist_")]].isna().any().any()
