import pandas as pd
from FIT_python.pipeline_individual_id.simple_baseline import collect_id_metrics, plot_bcr_comparison


def test_collect_id_metrics(tmp_path):
    sp1 = tmp_path / "sp1"
    sp2 = tmp_path / "sp2"
    sp1.mkdir()
    sp2.mkdir()
    pd.DataFrame({
        "bcr": [0.7, 0.9],
        "erd": [0.1, 0.2],
        "pred_count": [5, 5],
        "true_count": [5, 5],
        "ccc": [1.0, 1.0],
    }).to_json(sp1 / "summary.json", orient="records")
    pd.DataFrame({
        "bcr": [0.4, 0.6],
        "erd": [0.3, 0.4],
        "pred_count": [4, 4],
        "true_count": [5, 5],
        "ccc": [0.8, 0.8],
    }).to_json(sp2 / "summary.json", orient="records")

    df = collect_id_metrics(tmp_path)
    assert set(df["species"]) == {"sp1", "sp2"}
    assert (tmp_path / "raw_results.json").exists()
    s1 = df[df["species"] == "sp1"].iloc[0]
    assert abs(s1["bcr"] - 0.8) < 1e-6


def test_plot_bcr_comparison(tmp_path):
    df = pd.DataFrame({"species": ["a", "b"], "bcr": [0.5, 0.6]})
    out = plot_bcr_comparison(df, tmp_path)
    assert out.exists()


def test_load_splits_filters_unknown(tmp_path):
    sp = tmp_path / "sp"
    sp.mkdir()
    train = pd.DataFrame(
        {
            "Trail": ["t1", "unknown", None],
            "Fold": [0, 1, 1],
            "individual_id": ["A", "B", "C"],
            "sex": ["m", "f", "m"],
        }
    )
    train.to_parquet(sp / "train.parquet", index=False)
    test = pd.DataFrame(
        {
            "Trail": ["t2"],
            "Fold": [0],
            "individual_id": ["D"],
            "sex": ["m"],
        }
    )
    test.to_parquet(sp / "test.parquet", index=False)

    from FIT_python.pipeline_individual_id.simple_baseline import _load_splits

    df = _load_splits(sp, include_test=True)
    assert df["Trail"].notna().all()
    assert not df["Trail"].str.lower().eq("unknown").any()


def test_run_simple_baseline_all_species(tmp_path, monkeypatch):
    root = tmp_path
    sp_dir = root / "data" / "splits" / "sp"
    sp_dir.mkdir(parents=True)

    rows = []
    idx = 0
    for fold in [0, 1]:
        for ind in ["A", "B"]:
            rows.append(
                {
                    "id": idx,
                    "individual_id": ind,
                    "f1": float(idx),
                    "Trail": f"{ind}_{fold}",
                    "sex": "f" if ind == "A" else "m",
                    "fold": fold,
                }
            )
            idx += 1
    df = pd.DataFrame(rows)
    df.to_parquet(sp_dir / "train.parquet", index=False)

    monkeypatch.setenv("FIT_EXPERIMENT_ROOT", str(root))
    monkeypatch.setenv("FIT_RAW_DIR", str(root / "data" / "raw"))
    import FIT_python.config as cfg
    import importlib
    importlib.reload(cfg)
    from FIT_python.soft_config import SOFT_CONFIG

    SOFT_CONFIG["pipeline_individual_id"]["trail_generation_defaults"]["sample_size"] = 1

    import FIT_python.pipeline_individual_id.simple_baseline as sb
    importlib.reload(sb)
    from pathlib import Path

    sb.run_simple_baseline_all_species(
        root / "exp",
        best_k=1,
        cutoff={},
        subsample=True,
        reuse_summary=False,
        n_jobs=1,
    )

    assert (root / "exp" / "sp" / "summary.json").exists()


def test_run_simple_baseline_all_species_with_sex(tmp_path, monkeypatch):
    root = tmp_path
    sp_dir = root / "data" / "splits" / "sp"
    sp_dir.mkdir(parents=True)

    rows = []
    idx = 0
    for fold in [0, 1]:
        for ind in ["A", "B"]:
            rows.append(
                {
                    "id": idx,
                    "individual_id": ind,
                    "f1": float(idx),
                    "Trail": f"{ind}_{fold}",
                    "sex": "f" if ind == "A" else "m",
                    "fold": fold,
                }
            )
            idx += 1
    df = pd.DataFrame(rows)
    df.to_parquet(sp_dir / "train.parquet", index=False)

    monkeypatch.setenv("FIT_EXPERIMENT_ROOT", str(root))
    monkeypatch.setenv("FIT_RAW_DIR", str(root / "data" / "raw"))
    import FIT_python.config as cfg
    import importlib
    importlib.reload(cfg)
    from FIT_python.soft_config import SOFT_CONFIG

    SOFT_CONFIG["pipeline_individual_id"]["trail_generation_defaults"][
        "sample_size"
    ] = 1

    import FIT_python.pipeline_individual_id.simple_baseline as sb
    importlib.reload(sb)

    preds = pd.DataFrame({"id": [0], "pred_f_proba_f": [0.5]})
    calls = []

    metric = cfg.SEX_PREDICT_METRIC

    def fake_load(species, models_dir=None, metric_key=None):
        calls.append((species, models_dir, metric_key))
        return preds

    from pathlib import Path

    def fake_run(*args, sex_predictions=None, out_dir=None, tag=None, master_fp=None, **kwargs):
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        assert sex_predictions is preds
        assert master_fp == Path(out_dir) / "master_pairs.parquet"
        expected_tag = f"{SOFT_CONFIG['pipeline_individual_id']['pairwise_defaults']['selection_method']}_k1_sex_off"
        assert tag == expected_tag
        df_out = pd.DataFrame({"bcr": [1.0]})
        df_out.to_json(Path(out_dir) / "summary.json", orient="records")
        return df_out

    monkeypatch.setattr(sb, "load_sex_predictions", fake_load)
    monkeypatch.setattr(sb, "run_fold_cv", fake_run)

    sb.run_simple_baseline_all_species(
        root / "exp",
        best_k=1,
        cutoff={},
        subsample=True,
        reuse_summary=False,
        n_jobs=1,
        use_sex_predictions=True,
        models_dir=root / "models",
    )

    assert calls == [("sp", root / "models", metric)]
    assert (root / "exp" / "sp" / "summary.json").exists()


def test_run_simple_baseline_all_species_with_sexmodel(tmp_path, monkeypatch):
    root = tmp_path
    sp_dir = root / "data" / "splits" / "sp"
    sp_dir.mkdir(parents=True)

    rows = []
    idx = 0
    for fold in [0, 1]:
        for ind in ["A", "B"]:
            rows.append(
                {
                    "id": idx,
                    "individual_id": ind,
                    "f1": float(idx),
                    "Trail": f"{ind}_{fold}",
                    "sex": "f" if ind == "A" else "m",
                    "fold": fold,
                }
            )
            idx += 1
    pd.DataFrame(rows).to_parquet(sp_dir / "train.parquet", index=False)

    monkeypatch.setenv("FIT_EXPERIMENT_ROOT", str(root))
    monkeypatch.setenv("FIT_RAW_DIR", str(root / "data" / "raw"))
    import FIT_python.config as cfg
    import importlib
    importlib.reload(cfg)
    from FIT_python.soft_config import SOFT_CONFIG

    SOFT_CONFIG["pipeline_individual_id"]["trail_generation_defaults"]["sample_size"] = 1

    import FIT_python.pipeline_individual_id.simple_baseline as sb
    importlib.reload(sb)
    from pathlib import Path

    captured = {}

    def fake_run(*args, use_sexmodel_prediction=None, sexmodel_path=None, out_dir=None, **kwargs):
        from pathlib import Path
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        captured["use_sexmodel_prediction"] = use_sexmodel_prediction
        captured["sexmodel_path"] = sexmodel_path
        df_out = pd.DataFrame({"bcr": [1.0]})
        df_out.to_json(Path(out_dir) / "summary.json", orient="records")
        return df_out

    monkeypatch.setattr(sb, "run_fold_cv", fake_run)

    sb.run_simple_baseline_all_species(
        root / "exp",
        best_k=1,
        cutoff={},
        subsample=True,
        reuse_summary=False,
        n_jobs=1,
        use_sexmodel_prediction=True,
    )

    assert captured["use_sexmodel_prediction"] is True
    from FIT_python.utils import get_species_paths
    metric = cfg.SEX_PREDICT_METRIC
    expected = get_species_paths("sp")["models"] / metric / "sp.joblib"
    assert Path(captured["sexmodel_path"]) == expected
    assert (root / "exp" / "sp" / "summary.json").exists()


def test_run_simple_baseline_all_species_forward_params(tmp_path, monkeypatch):
    root = tmp_path
    sp_dir = root / "data" / "splits" / "sp"
    sp_dir.mkdir(parents=True)

    rows = []
    idx = 0
    for fold in [0, 1]:
        for ind in ["A", "B"]:
            rows.append(
                {
                    "id": idx,
                    "individual_id": ind,
                    "f1": float(idx),
                    "Trail": f"{ind}_{fold}",
                    "sex": "f" if ind == "A" else "m",
                    "fold": fold,
                }
            )
            idx += 1
    pd.DataFrame(rows).to_parquet(sp_dir / "train.parquet", index=False)

    monkeypatch.setenv("FIT_EXPERIMENT_ROOT", str(root))
    monkeypatch.setenv("FIT_RAW_DIR", str(root / "data" / "raw"))
    import FIT_python.config as cfg
    import importlib
    importlib.reload(cfg)
    from FIT_python.soft_config import SOFT_CONFIG

    SOFT_CONFIG["pipeline_individual_id"]["trail_generation_defaults"]["sample_size"] = 1

    import FIT_python.pipeline_individual_id.simple_baseline as sb
    importlib.reload(sb)
    from pathlib import Path

    captured = {}

    def fake_run(*args, selection_method=None, reducers=None, n_components=None, scaler_methods=None, out_dir=None, **kwargs):
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        captured["selection_method"] = selection_method
        captured["reducers"] = reducers
        captured["n_components"] = n_components
        captured["scaler_methods"] = scaler_methods
        df_out = pd.DataFrame({"bcr": [1.0]})
        df_out.to_json(Path(out_dir) / "summary.json", orient="records")
        return df_out

    monkeypatch.setattr(sb, "run_fold_cv", fake_run)

    sb.run_simple_baseline_all_species(
        root / "exp",
        best_k=1,
        cutoff={},
        subsample=True,
        reuse_summary=False,
        n_jobs=1,
        selection_method="foo",
        reducers=["pca"],
        n_components=3,
        scaler_methods="standard",
    )

    assert captured == {
        "selection_method": "foo",
        "reducers": ["pca"],
        "n_components": 3,
        "scaler_methods": "standard",
    }
    assert (root / "exp" / "sp" / "summary.json").exists()


def test_run_simple_baseline_all_species_none_n_components(tmp_path, monkeypatch):
    root = tmp_path
    sp_dir = root / "data" / "splits" / "sp"
    sp_dir.mkdir(parents=True)

    rows = []
    idx = 0
    for fold in [0, 1]:
        for ind in ["A", "B"]:
            rows.append(
                {
                    "id": idx,
                    "individual_id": ind,
                    "f1": float(idx),
                    "Trail": f"{ind}_{fold}",
                    "sex": "f" if ind == "A" else "m",
                    "fold": fold,
                }
            )
            idx += 1
    pd.DataFrame(rows).to_parquet(sp_dir / "train.parquet", index=False)

    monkeypatch.setenv("FIT_EXPERIMENT_ROOT", str(root))
    monkeypatch.setenv("FIT_RAW_DIR", str(root / "data" / "raw"))
    import FIT_python.config as cfg
    import importlib
    importlib.reload(cfg)
    from FIT_python.soft_config import SOFT_CONFIG

    SOFT_CONFIG["pipeline_individual_id"]["trail_generation_defaults"]["sample_size"] = 1

    import FIT_python.pipeline_individual_id.simple_baseline as sb
    importlib.reload(sb)
    from pathlib import Path

    captured = {}

    def fake_run(*args, n_components=None, out_dir=None, **kwargs):
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        captured["n_components"] = n_components
        df_out = pd.DataFrame({"bcr": [1.0]})
        df_out.to_json(Path(out_dir) / "summary.json", orient="records")
        return df_out

    monkeypatch.setattr(sb, "run_fold_cv", fake_run)

    sb.run_simple_baseline_all_species(
        root / "exp",
        best_k=1,
        cutoff={},
        subsample=True,
        reuse_summary=False,
        n_jobs=1,
        n_components=None,
    )

    assert captured["n_components"] is None
    assert (root / "exp" / "sp" / "summary.json").exists()


def test_run_baseline_all_species_forward_params(tmp_path, monkeypatch):
    root = tmp_path
    sp_dir = root / "data" / "splits" / "sp"
    sp_dir.mkdir(parents=True)

    rows = []
    idx = 0
    for fold in [0, 1]:
        for ind in ["A", "B"]:
            rows.append(
                {
                    "id": idx,
                    "individual_id": ind,
                    "f1": float(idx),
                    "Trail": f"{ind}_{fold}",
                    "sex": "f" if ind == "A" else "m",
                    "fold": fold,
                }
            )
            idx += 1
    pd.DataFrame(rows).to_parquet(sp_dir / "train.parquet", index=False)

    monkeypatch.setenv("FIT_EXPERIMENT_ROOT", str(root))
    monkeypatch.setenv("FIT_RAW_DIR", str(root / "data" / "raw"))
    import FIT_python.config as cfg
    import importlib
    importlib.reload(cfg)
    from FIT_python.soft_config import SOFT_CONFIG

    SOFT_CONFIG["pipeline_individual_id"]["trail_generation_defaults"]["sample_size"] = 1

    import FIT_python.pipeline_individual_id.simple_baseline as sb
    importlib.reload(sb)
    from pathlib import Path

    captured = {}

    def fake_run(
        *args,
        selection_method=None,
        reducers=None,
        n_components=None,
        scaler_methods=None,
        out_dir=None,
        **kwargs,
    ):
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        captured["selection_method"] = selection_method
        captured["reducers"] = reducers
        captured["n_components"] = n_components
        captured["scaler_methods"] = scaler_methods
        df_out = pd.DataFrame({"bcr": [1.0]})
        df_out.to_csv(Path(out_dir) / "summary.csv", index=False)
        return df_out

    monkeypatch.setattr(sb.sequential_holdout, "run", fake_run)

    sb.run_baseline_all_species(
        root / "exp",
        best_k=1,
        cutoff={},
        reuse_summary=False,
        n_jobs=1,
        selection_method="foo",
        reducers=["pca"],
        n_components=3,
        scaler_methods="standard",
    )

    assert captured == {
        "selection_method": "foo",
        "reducers": ["pca"],
        "n_components": 3,
        "scaler_methods": "standard",
    }
    assert (root / "exp" / "sp" / "summary.csv").exists()


def test_run_baseline_all_species_none_n_components(tmp_path, monkeypatch):
    root = tmp_path
    sp_dir = root / "data" / "splits" / "sp"
    sp_dir.mkdir(parents=True)

    rows = []
    idx = 0
    for fold in [0, 1]:
        for ind in ["A", "B"]:
            rows.append(
                {
                    "id": idx,
                    "individual_id": ind,
                    "f1": float(idx),
                    "Trail": f"{ind}_{fold}",
                    "sex": "f" if ind == "A" else "m",
                    "fold": fold,
                }
            )
            idx += 1
    pd.DataFrame(rows).to_parquet(sp_dir / "train.parquet", index=False)

    monkeypatch.setenv("FIT_EXPERIMENT_ROOT", str(root))
    monkeypatch.setenv("FIT_RAW_DIR", str(root / "data" / "raw"))
    import FIT_python.config as cfg
    import importlib
    importlib.reload(cfg)
    from FIT_python.soft_config import SOFT_CONFIG

    SOFT_CONFIG["pipeline_individual_id"]["trail_generation_defaults"]["sample_size"] = 1

    import FIT_python.pipeline_individual_id.simple_baseline as sb
    importlib.reload(sb)
    from pathlib import Path

    captured = {}

    def fake_run(*args, n_components=None, out_dir=None, **kwargs):
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        captured["n_components"] = n_components
        df_out = pd.DataFrame({"bcr": [1.0]})
        df_out.to_csv(Path(out_dir) / "summary.csv", index=False)
        return df_out

    monkeypatch.setattr(sb.sequential_holdout, "run", fake_run)

    sb.run_baseline_all_species(
        root / "exp",
        best_k=1,
        cutoff={},
        reuse_summary=False,
        n_jobs=1,
        n_components=None,
    )

    assert (
        captured["n_components"]
        == SOFT_CONFIG["pipeline_individual_id"]["pairwise_defaults"][
            "n_components"
        ]
    )
    assert (root / "exp" / "sp" / "summary.csv").exists()
