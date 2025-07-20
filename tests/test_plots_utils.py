import pandas as pd
from pathlib import Path

from FIT_python.Visualisations.plots_utils import plot_pair_examples, plot_dendrogram


def test_plot_pair_examples(tmp_path: Path):
    df = pd.DataFrame({
        'same_individual': [True, True, False, False],
        'pred': [True, False, True, False],
        'coords_a_x': [[0.0, 1.0]] * 4,
        'coords_a_y': [[0.0, 0.0]] * 4,
        'coords_b_x': [[1.0, 2.0]] * 4,
        'coords_b_y': [[0.0, 0.0]] * 4,
    })
    out = plot_pair_examples(df, tmp_path)
    assert out.exists()


def test_plot_dendrogram(tmp_path: Path):
    dist = pd.DataFrame(
        [[0.0, 1.0, 2.0], [1.0, 0.0, 3.0], [2.0, 3.0, 0.0]],
        index=list('ABC'),
        columns=list('ABC'),
    )
    out_file = tmp_path / 'den.png'
    out = plot_dendrogram(dist, 1.5, out_file)
    assert out.exists()
