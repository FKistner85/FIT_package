import pandas as pd
from pathlib import Path

from FIT_python.pipeline_sex.baseline_sex import (
    plot_accuracy_comparison,
    plot_majority_comparison,
    plot_accuracy_by_sex,
)


def test_plot_accuracy_accepts_short_columns(tmp_path: Path):
    df = pd.DataFrame({
        "species": ["a", "b"],
        "accuracy": [0.5, 0.6],
        "f1": [0.4, 0.5],
    })
    out = plot_accuracy_comparison(df, tmp_path)
    assert out.exists()


def test_plot_majority_accepts_short_column(tmp_path: Path):
    df = pd.DataFrame({"species": ["a", "b"], "maj_pct": [0.7, 0.8]})
    out = plot_majority_comparison(df, tmp_path)
    assert out.exists()


def test_plot_accuracy_by_sex(tmp_path: Path):
    df = pd.DataFrame(
        {
            "species": ["a", "b"],
            "female_individual_acc": [0.7, 0.8],
            "male_individual_acc": [0.6, 0.9],
        }
    )
    out = plot_accuracy_by_sex(df, tmp_path)
    assert out.exists()

