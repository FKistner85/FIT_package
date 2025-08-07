from pathlib import Path

import pandas as pd

from FIT_python.utils.aggregate_all_folds import aggregate_all_folds


def test_aggregate_all_folds_filters_origin(tmp_path):
    base = tmp_path
    sp1 = base / "sp1"
    sp2 = base / "sp2"
    sp1.mkdir()
    sp2.mkdir()

    df1 = pd.DataFrame({"x": [1, 2], "origin": ["cv", "test"]})
    df2 = pd.DataFrame({"x": [3], "origin": ["cv"]})
    df1.to_parquet(sp1 / "master_pairs.parquet", index=False)
    df2.to_parquet(sp2 / "master_pairs.parquet", index=False)

    df_cv = aggregate_all_folds(base, origin="cv")
    assert set(df_cv["origin"]) == {"cv"}
    assert set(df_cv["species"]) == {"sp1", "sp2"}
    assert (base / "all_species_folds.parquet").exists()

    df_test = aggregate_all_folds(base, origin="test")
    assert set(df_test["origin"]) == {"test"}
    assert set(df_test["species"]) == {"sp1"}
