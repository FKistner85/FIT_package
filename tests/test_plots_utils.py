import pandas as pd
from pathlib import Path

from FIT_python.Visualisations.plots_utils import (
    plot_pair_examples,
    plot_dendrogram,
    plot_sex_feature_boxplots,
    plot_pred_true_counts,
)


def test_plot_pair_examples(tmp_path: Path):
    df = pd.DataFrame(
        {
            "same_individual": [True, True, False, False],
            "pred": [True, False, True, False],
            "coords_a_x": [[0.0, 1.0]] * 4,
            "coords_a_y": [[0.0, 0.0]] * 4,
            "coords_b_x": [[1.0, 2.0]] * 4,
            "coords_b_y": [[0.0, 0.0]] * 4,
        }
    )
    out_paths = plot_pair_examples(df, tmp_path)
    assert len(out_paths) == 4
    for p in out_paths:
        assert p.exists()


def test_plot_pair_examples_rhombus(tmp_path: Path):
    df = pd.DataFrame(
        {
            "same_individual": [True, True, False, False],
            "pred": [True, False, True, False],
            "coords_a_x": [[0.0, 1.0]] * 4,
            "coords_a_y": [[0.0, 0.0]] * 4,
            "coords_b_x": [[1.0, 2.0]] * 4,
            "coords_b_y": [[0.0, 0.0]] * 4,
        }
    )
    out_paths = plot_pair_examples(df, tmp_path, rhombus=True)
    assert len(out_paths) == 4
    for p in out_paths:
        assert p.exists()


def test_plot_dendrogram(tmp_path: Path):
    dist = pd.DataFrame(
        [[0.0, 1.0, 2.0], [1.0, 0.0, 3.0], [2.0, 3.0, 0.0]],
        index=list("ABC"),
        columns=list("ABC"),
    )
    out_file = tmp_path / "den.png"
    out = plot_dendrogram(dist, out_file, cutoff_low=1.0, cutoff_high=2.0)
    assert out.exists()


def test_plot_sex_feature_boxplots(tmp_path: Path):
    df_with = pd.DataFrame(
        {
            "species": ["a", "a", "b", "b"],
            "bcr": [0.6, 0.7, 0.5, 0.6],
            "pred_count": [3, 4, 2, 3],
            "true_count": [3, 3, 2, 2],
        }
    )
    df_without = pd.DataFrame(
        {
            "species": ["a", "a", "b", "b"],
            "bcr": [0.5, 0.6, 0.4, 0.5],
            "pred_count": [3, 5, 2, 4],
            "true_count": [3, 3, 2, 2],
        }
    )
    paths = plot_sex_feature_boxplots(
        {"with_sex": df_with, "without_sex": df_without}, tmp_path
    )
    for p in paths:
        assert p.exists()


def test_plot_pred_true_counts(tmp_path: Path):
    df_mean = pd.DataFrame({"pred_count": [1, 2], "true_count": [1, 1]})
    df_med = pd.DataFrame({"pred_count": [2, 3], "true_count": [2, 2]})
    out = plot_pred_true_counts({"mean": df_mean, "median": df_med}, tmp_path / "scatter.png")
    assert out.exists()
    assert out.stat().st_size > 0
