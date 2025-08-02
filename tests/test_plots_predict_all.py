import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from FIT_python.pipeline_sex.sex_predict_and_visualisation import (
    plot_individual_probabilities,
    plot_confusion,
)


def test_plot_individual_probabilities(tmp_path, monkeypatch):
    # Avoid writing caption sidecar files during tests
    monkeypatch.setattr(
        "FIT_python.caption_utils.save_caption", lambda *args, **kwargs: None
    )
    df = pd.DataFrame(
        {
            "individual_id": [1, 1, 2, 2],
            "sex": ["f", "f", "m", "m"],
            "pred_proba_m": [0.1, 0.2, 0.8, 0.9],
            "__split__": ["train", "test", "train", "test"],
        }
    )
    plot_individual_probabilities(df, tmp_path)
    # Expect plots for train and test to be created
    files = list(tmp_path.glob("*individual_probabilities*.png"))
    assert files, "Expected probability plots to be created"


def test_plot_confusion(monkeypatch):
    # Prevent plots from blocking tests
    monkeypatch.setattr(plt, "show", lambda *args, **kwargs: None)
    df = pd.DataFrame(
        {
            "sex": ["f", "m", "f", "m"],
            "pred_sex": ["f", "m", "m", "f"],
            "__split__": ["train", "train", "test", "test"],
        }
    )
    plot_confusion(df)  # Should run without error
