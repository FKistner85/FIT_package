import pytest
pytest.importorskip("pandas")
import pandas as pd

torch = pytest.importorskip("torch")

from FIT_python.pipeline_individual_id.siamese_pipeline import run as run_siamese


def build_train_df():
    return pd.DataFrame({
        "id": [0, 1, 2, 3],
        "individual_id": ["A", "A", "B", "B"],
        "f1": [0.0, 0.1, 1.0, 1.1],
        "f2": [0.0, 0.1, 1.0, 1.1],
    })


def build_val_df():
    return pd.DataFrame({
        "id": [4, 5],
        "individual_id": ["A", "B"],
        "f1": [0.2, 1.2],
        "f2": [0.2, 1.2],
    })


def build_comparisons():
    return [
        {
            "trail_a_id": "valA",
            "trail_b_id": "trainB",
            "samples_a": [4],
            "samples_b": [2, 3],
            "same_individual": False,
            "fold": 0,
        }
    ]


def test_run_with_val_df():
    train_df = build_train_df()
    val_df = build_val_df()
    comps = build_comparisons()

    res = run_siamese(
        train_df,
        comps,
        ["f1", "f2"],
        epochs=1,
        batch_size=2,
        embedding_dim=2,
        hidden_dim=4,
        val_df=val_df,
    )

    assert len(res) == len(comps)
