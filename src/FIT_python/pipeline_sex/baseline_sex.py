"""Baseline sex-classification helper."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from FIT_python.config import SPLITS_DIR
from FIT_python.caption_utils import save_caption
from FIT_python.Visualisations.plot_style import apply_style
from .pipeline_wrapper_sex import PipelineWrapper


def run_baseline_all_species(exp_dir: Path) -> None:
    """Train baseline LDA classifiers for each species.

    Parameters
    ----------
    exp_dir:
        Directory where ``raw_results.csv`` and best models will be stored.
    """
    exp_dir = Path(exp_dir)
    exp_dir.mkdir(parents=True, exist_ok=True)
    model_dir = exp_dir / "models"
    model_dir.mkdir(exist_ok=True)
    best_dir = exp_dir / "best_models"
    best_dir.mkdir(exist_ok=True)

    species_dirs = [d for d in sorted(SPLITS_DIR.iterdir()) if d.is_dir()]
    if not species_dirs:
        raise FileNotFoundError(f"No split directories found in {SPLITS_DIR}")
    names = ", ".join(d.name for d in species_dirs)
    print(f"[INFO] Found species: {names}")

    wrapper = PipelineWrapper(model_keys=["lda"], fs_method="forward")
    wrapper._model_dir = model_dir
    wrapper._best_dir = best_dir

    wrapper.prepare()

    # PipelineWrapper internally iterates over ``data/splits``
    # and trains a model for each species directory found there.
    df = wrapper.train()

    df.to_csv(exp_dir / "raw_results.csv", index=False)

    return None


def collect_best_metrics(exp_dir: Path) -> pd.DataFrame:
    """Return metrics for the best accuracy model of each species.

    Parameters
    ----------
    exp_dir:
        Directory searched recursively for ``best_models.csv`` files.

    Returns
    -------
    pandas.DataFrame
        DataFrame with one row per species containing the metrics of the
        best model w.r.t ``accuracy_test``.
    """

    exp_dir = Path(exp_dir)
    csv_files = sorted(exp_dir.rglob("best_models.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No 'best_models.csv' found under {exp_dir}")

    rows: list[pd.Series] = []
    for csv in csv_files:
        df = pd.read_csv(csv)
        if df.empty:
            continue
        if "best_metric" in df.columns:
            sub = df[df["best_metric"] == "accuracy_test"]
            if sub.empty and "accuracy_test" in df.columns:
                sub = df.sort_values("accuracy_test", ascending=False).head(1)
        elif "accuracy_test" in df.columns:
            sub = df.sort_values("accuracy_test", ascending=False).head(1)
        else:
            continue
        row = sub.iloc[0]
        if "species" not in row.index:
            row["species"] = Path(csv).parent.name
        rows.append(row)

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows).reset_index(drop=True)


def plot_accuracy_comparison(df: pd.DataFrame, fig_dir: Path) -> Path:
    """Plot a bar chart comparing accuracy across species."""

    if "species" not in df.columns or "accuracy_test" not in df.columns:
        raise KeyError("DataFrame must contain 'species' and 'accuracy_test'")

    apply_style()
    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    metrics = {"accuracy_test": "Accuracy"}
    if "f1_test" in df.columns:
        metrics["f1_test"] = "F1"

    plot_df = df.melt(
        id_vars="species",
        value_vars=list(metrics.keys()),
        var_name="metric",
        value_name="value",
    )
    plot_df["metric"] = plot_df["metric"].map(metrics)

    order = df.sort_values("accuracy_test", ascending=False)["species"]

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(
        data=plot_df,
        x="species",
        y="value",
        hue="metric",
        order=order,
        ax=ax,
    )
    ax.set_xlabel("Species")
    ax.set_ylabel("Score")
    plt.xticks(rotation=45, ha="right")
    ax.legend(title="Metric")
    fig.tight_layout()

    out = fig_dir / "accuracy_comparison.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    save_caption(out, "Baseline accuracy per species")
    return out


def plot_majority_comparison(df: pd.DataFrame, fig_dir: Path) -> Path:
    """Plot the fraction of majority-correct individuals per species."""

    if "species" not in df.columns or "maj_test_pct" not in df.columns:
        raise KeyError("DataFrame must contain 'species' and 'maj_test_pct'")

    apply_style()
    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    order = df.sort_values("maj_test_pct", ascending=False)["species"]

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(
        data=df,
        x="species",
        y="maj_test_pct",
        order=order,
        ax=ax,
        color="#4C72B0",
        edgecolor="black",
    )
    ax.set_xlabel("Species")
    ax.set_ylabel("Majority correct")
    plt.xticks(rotation=45, ha="right")
    fig.tight_layout()

    out = fig_dir / "majority_comparison.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    save_caption(out, "Majority correct individuals per species")
    return out
