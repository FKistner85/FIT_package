#!/usr/bin/env python
"""Evaluate ellipse overlap predictions for individual ID results."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from FIT_python.pipeline_individual_id.evaluation import (
    compute_overlap_jsl_style,
    compute_confusion,
)
from FIT_python.Visualisations.plot_style import apply_style
from sklearn.metrics import precision_recall_fscore_support

# Probability grid to search
P_VALUES = np.linspace(0.5, 0.99, 10)


def _plot_confusion(cm: pd.DataFrame, out_path: Path) -> None:
    """Save a normalised confusion matrix heatmap to ``out_path``."""
    apply_style()
    fig, ax = plt.subplots(figsize=(4, 4))
    sns.heatmap(
        cm / cm.sum(axis=1, keepdims=True),
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=["Same", "Different"],
        yticklabels=["Same", "Different"],
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def evaluate(csv_path: Path, p_values: np.ndarray = P_VALUES) -> float:
    """Evaluate ``csv_path`` and store a confusion matrix plot next to it."""
    df = pd.read_csv(csv_path)

    metrics: list[tuple[float, float, float, float]] = []
    for p in p_values:
        preds = df.apply(compute_overlap_jsl_style, axis=1, p=p)
        prec, rec, f1, _ = precision_recall_fscore_support(
            df["same_individual"], preds, average="binary"
        )
        metrics.append((p, prec, rec, f1))

    best_p, best_prec, best_rec, best_f1 = max(metrics, key=lambda m: m[3])
    df["pred"] = df.apply(compute_overlap_jsl_style, axis=1, p=best_p)
    cm = compute_confusion(df, true_col="same_individual", pred_col="pred")
    out_png = csv_path.with_name("confusion_matrix.png")
    _plot_confusion(cm, out_png)
    print(
        f"{csv_path}: p={best_p:.3f} prec={best_prec:.3f} rec={best_rec:.3f} f1={best_f1:.3f}"
    )
    return best_f1


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate ellipse overlap predictions")
    ap.add_argument(
        "root",
        type=Path,
        nargs="?",
        default=Path("results/experiments/fit_start_to_finish/id_baseline"),
        help="Directory containing experiment results",
    )
    args = ap.parse_args()

    for csv_path in sorted(args.root.rglob("all_splits.csv")):
        evaluate(csv_path)


if __name__ == "__main__":
    main()
