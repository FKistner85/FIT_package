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
    }).to_csv(sp1 / "summary.csv", index=False)
    pd.DataFrame({
        "bcr": [0.4, 0.6],
        "erd": [0.3, 0.4],
        "pred_count": [4, 4],
        "true_count": [5, 5],
        "ccc": [0.8, 0.8],
    }).to_csv(sp2 / "summary.csv", index=False)

    df = collect_id_metrics(tmp_path)
    assert set(df["species"]) == {"sp1", "sp2"}
    assert (tmp_path / "raw_results.csv").exists()
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
    import FIT_python.config as cfg
    import importlib
    importlib.reload(cfg)
    from FIT_python.soft_config import SOFT_CONFIG

    SOFT_CONFIG["pipeline_individual_id"]["trail_generation_defaults"]["sample_size"] = 1

    import FIT_python.pipeline_individual_id.simple_baseline as sb
    importlib.reload(sb)

    sb.run_simple_baseline_all_species(
        root / "exp",
        best_k=1,
        cutoff={},
        reuse_summary=False,
        n_jobs=1,
    )

    assert (root / "exp" / "sp" / "summary.csv").exists()
