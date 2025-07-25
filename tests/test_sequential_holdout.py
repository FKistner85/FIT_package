import pandas as pd
import pytest
from FIT_python.pipeline_individual_id.sequential_holdout import run
from FIT_python.pipeline_individual_id.holdout_helper import generate_holdout_sets
from FIT_python.config import GLOBAL_RANDOM_SEED


def build_df():
    rows = []
    for i, ind in enumerate(['A', 'B', 'C']):
        for j in range(2):
            rows.append({
                'id': len(rows),
                'individual_id': ind,
                'f1': float(i),
                'f2': float(j),
                'trail': f"{ind}_{j}",
                'sex': 'F' if i % 2 == 0 else 'M',
                'fold': 0,
            })
    return pd.DataFrame(rows)


def test_run_produces_summary(tmp_path):
    df = build_df()
    preds = pd.DataFrame({'p_f': 0.5, 'p_m': 0.5}, index=df['id'])
    from FIT_python.soft_config import SOFT_CONFIG
    SOFT_CONFIG["pipeline_individual_id"]["trail_generation_defaults"]["sample_size"] = 1
    summary = run(df, ['f1', 'f2'], preds, val_sizes=[2], iterations=1, out_dir=tmp_path, n_jobs=1)
    assert not summary.empty
    assert 'bcr' in summary.columns
    assert 'erd' in summary.columns
    assert 'pred_count' in summary.columns
    assert 'true_count' in summary.columns
    assert 'ward_cutoff' in summary.columns
    assert 'cutoff_low' in summary.columns
    assert 'cutoff_high' in summary.columns
    assert 'ccc' in summary.columns
    assert (tmp_path / 'summary.csv').exists()


def test_generate_holdout_sets(tmp_path):
    df = build_df()
    species_dir = tmp_path / "otter"
    species_dir.mkdir()
    df.to_parquet(species_dir / "train.parquet", index=False)

    splits = generate_holdout_sets(tmp_path, val_sizes=[2], iterations=2, seed=GLOBAL_RANDOM_SEED)
    assert "otter" in splits
    assert len(splits["otter"]) == 2
    for split in splits["otter"]:
        train_ids = set(split["train_df"]["individual_id"].unique())
        val_ids = set(split["val_df"]["individual_id"].unique())
        assert train_ids.isdisjoint(val_ids)


def test_k_features_influences_results(tmp_path):
    """``run`` should yield different outcomes for different ``k_features``."""

    df = build_df()
    preds = pd.DataFrame({"p_f": 0.5, "p_m": 0.5}, index=df["id"])
    from FIT_python.soft_config import SOFT_CONFIG

    SOFT_CONFIG["pipeline_individual_id"]["trail_generation_defaults"]["sample_size"] = 1

    out1 = tmp_path / "k1"
    out2 = tmp_path / "k2"

    summary1 = run(
        df,
        ["f1", "f2"],
        preds,
        val_sizes=[2],
        iterations=1,
        out_dir=out1,
        n_jobs=1,
        k_features=1,
    )
    summary2 = run(
        df,
        ["f1", "f2"],
        preds,
        val_sizes=[2],
        iterations=1,
        out_dir=out2,
        n_jobs=1,
        k_features=2,
    )

    assert not summary1.empty and not summary2.empty
    if (out1 / "split_0.csv").exists() and (out2 / "split_0.csv").exists():
        df1 = pd.read_csv(out1 / "split_0.csv")
        df2 = pd.read_csv(out2 / "split_0.csv")
        if not df1.empty and not df2.empty:
            assert not df1.equals(df2)
            return

    assert (
        summary1.loc[0, "n_pairs"] != summary2.loc[0, "n_pairs"]
        or summary1.loc[0, "bcr"] != summary2.loc[0, "bcr"]
    )


def test_reuse_summary(tmp_path):
    df = build_df()
    from FIT_python.soft_config import SOFT_CONFIG

    SOFT_CONFIG["pipeline_individual_id"]["trail_generation_defaults"][
        "sample_size"
    ] = 1

    summary1 = run(
        df,
        ["f1", "f2"],
        val_sizes=[2],
        iterations=1,
        out_dir=tmp_path,
        n_jobs=1,
    )
    summary2 = run(
        df,
        ["f1", "f2"],
        val_sizes=[2],
        iterations=1,
        out_dir=tmp_path,
        n_jobs=1,
        reuse_summary=True,
    )

    pd.testing.assert_frame_equal(summary1, summary2)


def test_all_splits_csv(tmp_path):
    df = build_df()
    preds = pd.DataFrame({"p_f": 0.5, "p_m": 0.5}, index=df["id"])
    from FIT_python.soft_config import SOFT_CONFIG

    SOFT_CONFIG["pipeline_individual_id"]["trail_generation_defaults"][
        "sample_size"
    ] = 1

    run(
        df,
        ["f1", "f2"],
        preds,
        val_sizes=[2],
        iterations=2,
        out_dir=tmp_path,
        n_jobs=1,
    )

    all_csv = tmp_path / "all_splits.csv"
    assert all_csv.exists()
    df_all = pd.read_csv(all_csv)
    assert {"split", "iteration", "n_val"}.issubset(df_all.columns)

    csv_files = sorted(tmp_path.glob("split_*.csv"))
    total_rows = sum(len(pd.read_csv(fp)) for fp in csv_files)
    assert len(df_all) == total_rows


def test_evaluate_with_cutoff(tmp_path):
    from FIT_python.pipeline_individual_id.sequential_holdout import evaluate_with_cutoff

    rows = [
        {"trail_a_id": "t1", "trail_b_id": "t2", "ind_a": "A", "ind_b": "A", "dist_euclidean": 0.1, "split": 0},
        {"trail_a_id": "t1", "trail_b_id": "t3", "ind_a": "A", "ind_b": "B", "dist_euclidean": 0.9, "split": 0},
        {"trail_a_id": "t2", "trail_b_id": "t3", "ind_a": "A", "ind_b": "B", "dist_euclidean": 0.8, "split": 0},
        {"trail_a_id": "t4", "trail_b_id": "t5", "ind_a": "C", "ind_b": "C", "dist_euclidean": 0.2, "split": 1},
        {"trail_a_id": "t4", "trail_b_id": "t6", "ind_a": "C", "ind_b": "D", "dist_euclidean": 0.9, "split": 1},
        {"trail_a_id": "t5", "trail_b_id": "t6", "ind_a": "C", "ind_b": "D", "dist_euclidean": 0.8, "split": 1},
    ]
    df = pd.DataFrame(rows)
    df.to_csv(tmp_path / "all_splits.csv", index=False)

    res = evaluate_with_cutoff(tmp_path, cutoff=0.5)
    assert list(res["pred_count"]) == [2, 2]
    assert list(res["true_count"]) == [2, 2]
    assert list(res["erd"]) == [0.0, 0.0]


def test_compute_global_cutoffs(tmp_path):
    from FIT_python.pipeline_individual_id.sequential_holdout import (
        compute_global_cutoffs,
    )

    rows = [
        {"trail_a_id": "t1", "trail_b_id": "t2", "ind_a": "A", "ind_b": "A", "dist_euclidean": 0.1, "split": 0},
        {"trail_a_id": "t1", "trail_b_id": "t3", "ind_a": "A", "ind_b": "B", "dist_euclidean": 0.9, "split": 0},
        {"trail_a_id": "t2", "trail_b_id": "t3", "ind_a": "A", "ind_b": "B", "dist_euclidean": 0.8, "split": 0},
        {"trail_a_id": "t4", "trail_b_id": "t5", "ind_a": "C", "ind_b": "C", "dist_euclidean": 0.2, "split": 1},
        {"trail_a_id": "t4", "trail_b_id": "t6", "ind_a": "C", "ind_b": "D", "dist_euclidean": 0.9, "split": 1},
        {"trail_a_id": "t5", "trail_b_id": "t6", "ind_a": "C", "ind_b": "D", "dist_euclidean": 0.8, "split": 1},
    ]
    df = pd.DataFrame(rows)
    csv = tmp_path / "all_splits.csv"
    df.to_csv(csv, index=False)

    stats = compute_global_cutoffs(csv)

    assert stats["mean_cutoff"] == pytest.approx(0.56447084, rel=1e-6)
    assert stats["median_cutoff"] == pytest.approx(0.56447084, rel=1e-6)
    assert stats["mean_low"] == pytest.approx(0.35723542, rel=1e-6)
    assert stats["mean_high"] == pytest.approx(0.77170626, rel=1e-6)
