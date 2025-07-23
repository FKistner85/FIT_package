import pandas as pd
from pathlib import Path

from FIT_python.Visualisations.plots_utils import (
    plot_pair_examples,
    plot_dendrogram,
    plot_sex_feature_boxplots,
    plot_pred_true_counts,
    plot_umap_by_individual,
    plot_embedding_by_individual,
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


def test_plot_dendrogram(tmp_path: Path, monkeypatch):
    from FIT_python.Visualisations import plots_utils as pu

    dist = pd.DataFrame(
        [[0.0, 1.0, 2.0], [1.0, 0.0, 3.0], [2.0, 3.0, 0.0]],
        index=["t1", "t2", "t3"],
        columns=["t1", "t2", "t3"],
    )

    monkeypatch.setattr(pu, "ID_COLORS", {"A": "red", "B": "blue"})
    monkeypatch.setattr(pu, "TRAIL_TO_ID", {"t1": "A", "t2": "B", "t3": "A"})

    captured = []

    def fake_close(fig=None):
        captured.append(fig)

    monkeypatch.setattr(pu.plt, "close", fake_close)

    out_file = tmp_path / "den.png"
    out = pu.plot_dendrogram(dist, out_file, cutoff_low=1.0, cutoff_high=2.0)
    assert out.exists()
    fig = captured[0]
    colors = [lab.get_color() for lab in fig.axes[0].get_xticklabels()]
    assert colors == ["red", "blue", "red"]


def test_plot_dendrogram_display_trail(tmp_path: Path, monkeypatch):
    from FIT_python.Visualisations import plots_utils as pu

    dist = pd.DataFrame(
        [[0.0, 1.0], [1.0, 0.0]],
        index=["t1", "t2"],
        columns=["t1", "t2"],
    )
    dist["display_trail"] = ["d1", "d2"]

    monkeypatch.setattr(pu, "ID_COLORS", {"A": "red", "B": "green"})
    monkeypatch.setattr(pu, "TRAIL_TO_ID", {"t1": "A", "t2": "B"})

    captured = []

    def fake_close(fig=None):
        captured.append(fig)

    monkeypatch.setattr(pu.plt, "close", fake_close)

    out_file = tmp_path / "den.png"
    out = pu.plot_dendrogram(dist, out_file)
    assert out.exists()
    fig = captured[0]
    xticks = fig.axes[0].get_xticklabels()
    labels = [lab.get_text() for lab in xticks]
    colors = [lab.get_color() for lab in xticks]
    assert labels == ["d1", "d2"]
    assert colors == ["red", "green"]


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
    out = plot_pred_true_counts(
        {"mean": df_mean, "median": df_med},
        tmp_path / "scatter.png",
        regression=True,
        n_train=5,
    )
    assert out.exists()
    assert out.stat().st_size > 0


def test_umap_colors_use_id_palette(tmp_path: Path, monkeypatch):
    from FIT_python.Visualisations import plots_utils as pu

    monkeypatch.setattr(pu, "ID_COLORS", {"A": "red", "B": "blue"})
    monkeypatch.setattr(pu, "ID_MARKERS", {"A": "o", "B": "s"})

    captured = []

    def fake_scatter(ax, x, y, color, marker, label=""):
        captured.append((color, marker))

    monkeypatch.setattr(pu, "_scatter_points", fake_scatter)

    df = pd.DataFrame({"individual_id": ["A", "B"], "UMAP1": [0, 1], "UMAP2": [1, 2]})
    out = pu.plot_umap_by_individual(df, tmp_path, "out.png")
    assert out.exists()
    assert captured == [("red", "o"), ("blue", "s")]


def test_umap_colors_use_sex_palette(tmp_path: Path, monkeypatch):
    from FIT_python.Visualisations import plots_utils as pu

    monkeypatch.setattr(pu, "SEX_COLORS", {"Female": "pink", "Male": "cyan"})
    monkeypatch.setattr(pu, "ID_COLORS", {"A": "red", "B": "blue"})
    monkeypatch.setattr(pu, "ID_MARKERS", {"A": "o", "B": "s"})

    captured = []

    def fake_scatter(ax, x, y, color, marker, label=""):
        captured.append((color, marker))

    monkeypatch.setattr(pu, "_scatter_points", fake_scatter)

    df = pd.DataFrame(
        {
            "individual_id": ["A", "B"],
            "sex_mapped": ["Female", "Male"],
            "UMAP1": [0, 1],
            "UMAP2": [1, 2],
        }
    )
    out = pu.plot_umap_by_individual(df, tmp_path, "sex.png", use_sex_colors=True)
    assert out.exists()
    assert captured == [("pink", "o"), ("cyan", "s")]


def test_umap_scatter_legend_flag(tmp_path: Path, monkeypatch):
    from FIT_python.Visualisations import plots_utils as pu
    import matplotlib.axes

    calls = []

    orig_legend = matplotlib.axes.Axes.legend

    def fake_legend(self, *args, **kwargs):
        calls.append(True)
        return orig_legend(self, *args, **kwargs)

    monkeypatch.setattr(matplotlib.axes.Axes, "legend", fake_legend)

    emb = pd.DataFrame({"UMAP1": [0, 1], "UMAP2": [1, 2], "sex_mapped": ["Female", "Male"]})

    pu.plot_umap_scatter(pd.DataFrame(), emb, tmp_path, "a.png", legend=False)
    assert calls == []

    pu.plot_umap_scatter(pd.DataFrame(), emb, tmp_path, "b.png", legend=True)
    assert len(calls) >= 1


def test_umap_by_individual_legend_flag(tmp_path: Path, monkeypatch):
    from FIT_python.Visualisations import plots_utils as pu
    import matplotlib.axes

    calls = []

    orig_legend = matplotlib.axes.Axes.legend

    def fake_legend(self, *args, **kwargs):
        calls.append(True)
        return orig_legend(self, *args, **kwargs)

    monkeypatch.setattr(matplotlib.axes.Axes, "legend", fake_legend)

    df = pd.DataFrame({"individual_id": ["A", "B"], "UMAP1": [0, 1], "UMAP2": [1, 2]})

    pu.plot_umap_by_individual(df, tmp_path, "c.png", legend=False)
    assert calls == []

    pu.plot_umap_by_individual(df, tmp_path, "d.png", legend=True)
    assert calls[-1:] == [True]


def test_embedding_colors_use_id_palette(tmp_path: Path, monkeypatch):
    from FIT_python.Visualisations import plots_utils as pu

    monkeypatch.setattr(pu, "ID_COLORS", {"A": "red", "B": "blue"})
    monkeypatch.setattr(pu, "ID_MARKERS", {"A": "o", "B": "s"})

    captured = []

    def fake_scatter(ax, x, y, color, marker, label=""):
        captured.append((color, marker))

    monkeypatch.setattr(pu, "_scatter_points", fake_scatter)

    train_df = pd.DataFrame({"individual_id": ["A"], "x": [0], "y": [0]})
    test_df = pd.DataFrame({"individual_id": ["B"], "x": [1], "y": [1]})
    out = pu.plot_embedding_by_individual(
        train_df,
        test_df,
        tmp_path,
        "emb.png",
        "x",
        "y",
        ("Train", "Test"),
    )
    assert out.exists()
    assert captured == [("red", "o"), ("blue", "s")]


def test_umap_by_individual_display_labels(tmp_path: Path, monkeypatch):
    from FIT_python.Visualisations import plots_utils as pu

    monkeypatch.setattr(pu, "ID_COLORS", {"A": "red"})
    monkeypatch.setattr(pu, "ID_MARKERS", {"A": "o"})

    labels = []

    def fake_scatter(ax, x, y, color, marker, label=""):
        labels.append(label)

    monkeypatch.setattr(pu, "_scatter_points", fake_scatter)

    df = pd.DataFrame({"individual_id": ["A"], "UMAP1": [0], "UMAP2": [1]})
    pu.plot_umap_by_individual(df, tmp_path, "out_map.png", mapping={"A": "Ind_1"})
    assert labels == ["Ind_1"]


def test_embedding_display_labels(tmp_path: Path, monkeypatch):
    from FIT_python.Visualisations import plots_utils as pu

    monkeypatch.setattr(pu, "ID_COLORS", {"A": "red"})
    monkeypatch.setattr(pu, "ID_MARKERS", {"A": "o"})

    labels = []

    def fake_scatter(ax, x, y, color, marker, label=""):
        labels.append(label)

    monkeypatch.setattr(pu, "_scatter_points", fake_scatter)

    train_df = pd.DataFrame({"individual_id": ["A"], "x": [0], "y": [0]})
    test_df = pd.DataFrame({"individual_id": ["A"], "x": [1], "y": [1]})
    pu.plot_embedding_by_individual(
        train_df,
        test_df,
        tmp_path,
        "emb2.png",
        "x",
        "y",
        ("Train", "Test"),
        mapping={"A": "Ind_1"},
    )
    assert labels[-1] == "Ind_1"
