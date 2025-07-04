import subprocess
import sys
import os
from pathlib import Path

import pandas as pd


def create_split(path: Path, sexes):
    df = pd.DataFrame({"sex": sexes})
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)


def test_summary_cli(tmp_path):
    splits = tmp_path / "splits"
    create_split(splits / "demo_train.parquet", ["F", "M", "F"])
    create_split(splits / "demo_test.parquet", ["M", "M"])
    create_split(splits / "demo_inference.parquet", ["F", "M"])

    out_csv = tmp_path / "summary.csv"
    fig_dir = tmp_path / "figs"

    script = Path(__file__).resolve().parents[1] / "scripts" / "summary.py"
    env = {
        **os.environ,
        "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src"),
    }
    subprocess.run(
        [sys.executable, str(script),
         "--splits-dir", str(splits),
         "--output-table", str(out_csv),
         "--fig-dir", str(fig_dir)],
        check=True,
        env=env,
    )

    assert out_csv.exists()
    df = pd.read_csv(out_csv)

    expected = {
        ("demo", origin.capitalize(), sex)
        for origin in ["train", "test", "inference"]
        for sex in ["F", "M", "Unknown"]
    }
    combos = set(zip(df["Dataset"], df["Origin"], df["Sex"]))
    assert expected == combos

    plot_file = fig_dir / "demo_summary.png"
    assert plot_file.exists() and plot_file.stat().st_size > 0
