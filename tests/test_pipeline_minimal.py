import pandas as pd
import numpy as np
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import FIT_python.config as config


def setup_dummy_dataset(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    df = pd.DataFrame({
        "individual_id": [1,1,2,2,3,3,4,4],
        "sex": ["F","F","M","M","F","M","F","M"],
        "feat1": range(8),
        "feat2": range(10,18),
        "feat3": np.linspace(0,1,8)
    })
    df.to_csv(raw_dir / "demo.csv", index=False)
    return raw_dir


def patch_config(tmp_path, monkeypatch):
    proc = tmp_path / "processed"
    monkeypatch.setattr(config, "RAW_DIR", tmp_path / "raw")
    monkeypatch.setattr(config, "PROCESSED_SPLITS_DIR", proc / "splits")
    monkeypatch.setattr(config, "SCALED_DIR", proc / "scaled")
    monkeypatch.setattr(config, "FEATURE_SELECTED_DIR", proc / "feature_selected")
    monkeypatch.setattr(config, "DIM_REDUCED_DIR", proc / "dim_reduced")
    monkeypatch.setattr(config, "DEFAULT_TARGETS", ["individual_id", "sex"])


def test_full_pipeline(tmp_path, monkeypatch):
    raw_dir = setup_dummy_dataset(tmp_path)
    patch_config(tmp_path, monkeypatch)

    import importlib
    import create_splits
    import scale_splits
    import create_feature_selection
    import create_dim_reduction
    importlib.reload(create_splits)
    importlib.reload(scale_splits)
    importlib.reload(create_feature_selection)
    importlib.reload(create_dim_reduction)

    create_splits.main()
    train_path = config.PROCESSED_SPLITS_DIR / "demo_train.parquet"
    test_path = config.PROCESSED_SPLITS_DIR / "demo_test.parquet"
    assert train_path.exists() and test_path.exists()

    scale_splits.main()
    scaled_train = pd.read_parquet(config.SCALED_DIR / "demo_train.parquet")
    feats = ["feat1", "feat2", "feat3"]
    assert np.allclose(scaled_train[feats].mean(), 0, atol=1e-6)
    assert np.allclose(scaled_train[feats].std(ddof=0), 1, atol=1e-6)

    create_feature_selection.main()
    fs_train = pd.read_parquet(config.FEATURE_SELECTED_DIR / "demo_train.parquet")
    cols = set(fs_train.columns)
    expected_feats = {"feat1", "feat2"}
    assert expected_feats.issubset(cols)
    assert "feat3" not in cols

    create_dim_reduction.main()
    red_path = config.DIM_REDUCED_DIR / "demo_train.parquet"
    assert red_path.exists()
    df_red = pd.read_parquet(red_path)
    assert df_red.shape[1] == 2
