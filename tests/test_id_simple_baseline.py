import pandas as pd
from FIT_python.pipeline_individual_id.simple_baseline import collect_id_metrics, plot_bcr_comparison


def test_collect_id_metrics(tmp_path):
    sp1 = tmp_path / "sp1"
    sp2 = tmp_path / "sp2"
    sp1.mkdir()
    sp2.mkdir()
    pd.DataFrame({
        "bcr": [0.7, 0.9],
        "erd": [0.1, 0.2],
        "pred_count": [5, 5],
        "true_count": [5, 5],
        "ccc": [1.0, 1.0],
    }).to_csv(sp1 / "summary.csv", index=False)
    pd.DataFrame({
        "bcr": [0.4, 0.6],
        "erd": [0.3, 0.4],
        "pred_count": [4, 4],
        "true_count": [5, 5],
        "ccc": [0.8, 0.8],
    }).to_csv(sp2 / "summary.csv", index=False)

    df = collect_id_metrics(tmp_path)
    assert set(df["species"]) == {"sp1", "sp2"}
    assert (tmp_path / "raw_results.csv").exists()
    s1 = df[df["species"] == "sp1"].iloc[0]
    assert abs(s1["bcr"] - 0.8) < 1e-6


def test_plot_bcr_comparison(tmp_path):
    df = pd.DataFrame({"species": ["a", "b"], "bcr": [0.5, 0.6]})
    out = plot_bcr_comparison(df, tmp_path)
    assert out.exists()
