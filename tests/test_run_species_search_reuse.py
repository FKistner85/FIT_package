import pandas as pd
import ast
from pathlib import Path


def _load_functions(tmp_root):
    sc_path = Path(__file__).resolve().parents[1] / "src" / "FIT_python" / "pipeline_sex" / "sex_config.py"
    source = sc_path.read_text()
    module = ast.parse(source)
    funcs = [n for n in module.body if isinstance(n, ast.FunctionDef) and n.name in {"_run_species_search", "run_species_search"}]
    code = compile(ast.Module(body=funcs, type_ignores=[]), sc_path.as_posix(), "exec")
    env = {
        "pd": pd,
        "Path": Path,
        "PIPE_CFG": {"run_otter_search_sex": {"n_iter": 1, "cv": 2, "random_state": 0}},
    }
    exec(code, env)
    return env["_run_species_search"], env["run_species_search"], env


def test__run_species_search_reuse(tmp_path, monkeypatch):
    root = tmp_path
    results_dir = root / "results" / "data" / "otter_bayes_search_standard_metrics"
    results_dir.mkdir(parents=True)
    df_all = pd.DataFrame({"a": [1]})
    df_best = pd.DataFrame({"b": [2]})
    df_all.to_csv(results_dir / "all_results.csv", index=False)
    df_best.to_csv(results_dir / "best_models.csv", index=False)
    (results_dir / "best_mean_rank" / "otter.joblib").parent.mkdir(parents=True)
    (results_dir / "best_mean_rank" / "otter.joblib").write_bytes(b"0")

    _run_species_search, _, env = _load_functions(root)
    g = env

    called = {"flag": False}

    class DummySearch:
        def __init__(self, *a, **k):
            called["flag"] = True
        def fit(self, X, y):
            called["flag"] = True

    g.update({
        "BayesSearchCV": DummySearch,
        "pd": pd,
        "RESULTS_DATA_DIR": root / "results" / "data",
        "SPLITS_DIR": root / "data" / "splits",
    })

    got_all, got_best = _run_species_search(
        "otter",
        "otter_bayes_search_standard_metrics",
        1,
        2,
        0,
        reuse_results=True,
    )

    assert not called["flag"]
    pd.testing.assert_frame_equal(got_all, df_all)
    pd.testing.assert_frame_equal(got_best, df_best)


def test_run_species_search_reuse(tmp_path, monkeypatch):
    root = tmp_path
    (root / "data" / "splits" / "otter").mkdir(parents=True)
    results_dir = root / "results" / "data" / "otter_bayes_search_standard_metrics"
    results_dir.mkdir(parents=True)
    df_all = pd.DataFrame({"a": [1]})
    df_best = pd.DataFrame({"b": [2]})
    df_all.to_csv(results_dir / "all_results.csv", index=False)
    df_best.to_csv(results_dir / "best_models.csv", index=False)

    _run_species_search, run_species_search, env = _load_functions(root)

    called = {"flag": False}

    class DummySearch:
        def __init__(self, *a, **k):
            called["flag"] = True
        def fit(self, X, y):
            called["flag"] = True

    env.update({
        "BayesSearchCV": DummySearch,
        "pd": pd,
        "RESULTS_DATA_DIR": root / "results" / "data",
        "SPLITS_DIR": root / "data" / "splits",
    })
    env["_run_species_search"] = _run_species_search

    got_all, got_best = run_species_search(species_filter=["otter"], reuse_results=True)

    assert not called["flag"]
    pd.testing.assert_frame_equal(got_all, df_all)
    pd.testing.assert_frame_equal(got_best, df_best)
