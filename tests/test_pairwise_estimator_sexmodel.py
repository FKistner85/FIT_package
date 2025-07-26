import importlib
import os
import pandas as pd


def test_search_space_contains_use_sexmodel_prediction(tmp_path, monkeypatch):
    monkeypatch.setenv("FIT_EXPERIMENT_ROOT", str(tmp_path))
    monkeypatch.setenv("FIT_RAW_DIR", str(tmp_path / "raw"))
    import FIT_python.config as cfg
    importlib.reload(cfg)
    ic = importlib.import_module("FIT_python.pipeline_individual_id.id_config")
    assert "est__use_sexmodel_prediction" in ic.SEARCH_SPACES
    cats = list(ic.SEARCH_SPACES["est__use_sexmodel_prediction"].categories)
    assert cats == [False, True]


def test_pairwise_estimator_passes_sexmodel_args(tmp_path, monkeypatch):
    monkeypatch.setenv("FIT_EXPERIMENT_ROOT", str(tmp_path))
    monkeypatch.setenv("FIT_RAW_DIR", str(tmp_path / "raw"))
    import FIT_python.config as cfg
    importlib.reload(cfg)
    ic = importlib.import_module("FIT_python.pipeline_individual_id.id_config")
    def fake_gen(_df):
        return [], None

    captured = {}

    def fake_run(comps, df, **kwargs):
        captured.update(kwargs)
        return []

    monkeypatch.setattr(ic, "generate_pairwise_comparisons_from_df", fake_gen)
    monkeypatch.setattr(ic, "run_all_pairwise_projections_parallel", fake_run)

    est = ic.PairwiseEstimator(
        ["f1"], use_sexmodel_prediction=True, sexmodel_path="model.joblib"
    )
    train = pd.DataFrame({"f1": [0], "individual_id": ["A"], "Trail": ["t1"], "id": [0]})
    est.fit(train)
    est.predict(train)
    assert captured["use_sexmodel_prediction"] is True
    assert captured["sexmodel_path"] == "model.joblib"
