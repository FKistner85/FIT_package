import pandas as pd
from pathlib import Path

from FIT_python.pipeline_sex.baseline_sex import (
    plot_accuracy_comparison,
    plot_majority_comparison,
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

