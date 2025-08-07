import importlib

from sklearn.base import clone


def test_pairwise_estimator_can_be_cloned(tmp_path, monkeypatch):
    monkeypatch.setenv("FIT_EXPERIMENT_ROOT", str(tmp_path))
    monkeypatch.setenv("FIT_RAW_DIR", str(tmp_path / "raw"))
    import FIT_python.config as cfg

    importlib.reload(cfg)
    ic = importlib.import_module("FIT_python.pipeline_individual_id.search")
    est = ic.PairwiseEstimator(["f1", "f2"])
    cloned = clone(est)
    assert isinstance(cloned, ic.PairwiseEstimator)
