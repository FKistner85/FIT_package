import pandas as pd
from FIT_python.pipeline_individual_id.sequential_holdout import run
from FIT_python.pipeline_individual_id.holdout_helper import generate_holdout_sets


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
    assert 'ccc' in summary.columns
    assert (tmp_path / 'summary.csv').exists()


def test_generate_holdout_sets(tmp_path):
    df = build_df()
    species_dir = tmp_path / "otter"
    species_dir.mkdir()
    df.to_parquet(species_dir / "train.parquet", index=False)

    splits = generate_holdout_sets(tmp_path, val_sizes=[2], iterations=2, seed=0)
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
