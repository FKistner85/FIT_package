import pandas as pd
import importlib
from pathlib import Path


def build_df():
    rows = []
    for i, ind in enumerate(["A", "B", "C"]):
        for j in range(2):
            rows.append(
                {
                    "id": len(rows),
                    "individual_id": ind,
                    "f1": float(i),
                    "Trail": f"{ind}_{j}",
                    "sex": "f" if i % 2 == 0 else "m",
                    "fold": 0,
                }
            )
    return pd.DataFrame(rows)


def test_sex_prediction_experiment_reuse(tmp_path, monkeypatch):
    root = tmp_path
    data_dir = root / "data" / "splits" / "otter"
    data_dir.mkdir(parents=True)
    df = build_df()
    df.to_parquet(data_dir / "train.parquet", index=False)
    df.to_parquet(data_dir / "test.parquet", index=False)

    preds_dir = root / "results" / "data" / "random_search_standard_metrics"
    preds_dir.mkdir(parents=True)
    pd.DataFrame({"id": [0], "pred_dummy_proba_f": [0.5]}).to_csv(
        preds_dir / "otter_all_predictions.csv", index=False
    )

    monkeypatch.setenv("FIT_EXPERIMENT_ROOT", str(root))
    import FIT_python.config as cfg

    importlib.reload(cfg)
    from FIT_python.soft_config import SOFT_CONFIG

    SOFT_CONFIG["pipeline_individual_id"]["trail_generation_defaults"][
        "sample_size"
    ] = 1

    import FIT_python.pipeline_individual_id.simple_baseline as sb

    importlib.reload(sb)

    def fake_run(*args, out_dir=None, **kwargs):
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        df_out = pd.DataFrame({"bcr": [1.0]})
        df_out.to_csv(Path(out_dir) / "summary.csv", index=False)
        return df_out

    monkeypatch.setattr(sb.sequential_holdout, "run", fake_run)

    exp_dir = root / "exp"
    sb.run_sex_prediction_experiment(
        exp_dir, best_k=1, cutoff={}, models_dir=preds_dir, reuse_results=False
    )

    with_df = pd.read_csv(exp_dir / "otter" / "with_sex" / "summary.csv")
    without_df = pd.read_csv(exp_dir / "otter" / "without_sex" / "summary.csv")

    sb.run_sex_prediction_experiment(
        exp_dir, best_k=1, cutoff={}, models_dir=preds_dir, reuse_results=True
    )

    pd.testing.assert_frame_equal(
        with_df, pd.read_csv(exp_dir / "otter" / "with_sex" / "summary.csv")
    )
    pd.testing.assert_frame_equal(
        without_df, pd.read_csv(exp_dir / "otter" / "without_sex" / "summary.csv")
    )
