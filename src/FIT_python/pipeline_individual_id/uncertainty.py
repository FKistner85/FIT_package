from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score

from FIT_python.analysis.uncertainty import (
    assign_confidence_bands,
    bootstrap_confidence_interval,
    build_calibration_table,
    expected_calibration_error,
)


def _to_binary(series: pd.Series) -> np.ndarray:
    """Return ``series`` encoded as a 0/1 numpy array."""

    if series.dtype == bool:
        return series.astype(int).to_numpy()
    if np.issubdtype(series.dtype, np.integer):
        arr = series.astype(int)
        if not set(arr.unique()).issubset({0, 1}):
            raise ValueError("Target column must only contain 0/1 values")
        return arr.to_numpy()
    if np.issubdtype(series.dtype, np.floating):
        arr = series.round().astype(int)
        if not set(arr.unique()).issubset({0, 1}):
            raise ValueError("Target column must only contain 0/1 values")
        return arr.to_numpy()

    lowered = series.astype(str).str.strip().str.lower()
    mapping = {
        "true": 1,
        "false": 0,
        "1": 1,
        "0": 0,
        "yes": 1,
        "no": 0,
        "same": 1,
        "different": 0,
    }
    unknown = set(lowered.unique()) - set(mapping)
    if unknown:
        raise ValueError(
            "Encountered unsupported target values: " + ", ".join(sorted(unknown))
        )
    return lowered.map(mapping).astype(int).to_numpy()


def summarise_pairwise_uncertainty(
    df: pd.DataFrame,
    *,
    proba_col: str = "pred_same_proba",
    target_col: str = "same_individual",
    group_cols: Sequence[str] | None = None,
    confidence_boundaries: Sequence[float] = (0.55, 0.7, 0.85),
    bootstrap_iterations: int = 500,
    bootstrap_confidence: float = 0.95,
    random_state: int | None = None,
    out_dir: str | Path | None = None,
    uncertain_threshold: float | None = None,
) -> dict[str, pd.DataFrame]:
    """Return calibration and confidence summaries for pairwise ID predictions."""

    if proba_col not in df.columns:
        raise KeyError(f"Probability column '{proba_col}' not found")
    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not found")

    group_cols = list(group_cols) if group_cols else []

    data = df.copy()
    proba = data[proba_col].astype(float).to_numpy()
    _ = _to_binary(data[target_col])  # validation step
    max_conf = np.maximum(proba, 1 - proba)
    data["max_confidence"] = max_conf

    boundaries = tuple(confidence_boundaries) if confidence_boundaries else tuple()
    if boundaries:
        data["confidence_band"] = assign_confidence_bands(
            data["max_confidence"], boundaries=boundaries
        )

    metrics_rows: list[dict[str, float]] = []
    calibration_rows: list[pd.DataFrame] = []
    confidence_rows: list[pd.DataFrame] = []

    log_loss_fn = lambda y, p: log_loss(
        y,
        np.clip(p, 1e-12, 1 - 1e-12),
        labels=[0, 1],
    )

    if group_cols:
        grouped = data.groupby(group_cols, dropna=False, sort=False)
    else:
        grouped = [((), data)]

    for group_idx, (keys, group_df) in enumerate(grouped):
        if group_cols:
            if not isinstance(keys, tuple):
                keys = (keys,)
            group_values = dict(zip(group_cols, keys))
        else:
            group_values = {"group": "all"}

        y_true = _to_binary(group_df[target_col])
        proba_split = group_df[proba_col].astype(float).to_numpy()
        max_conf_split = group_df["max_confidence"].to_numpy()
        y_pred = (proba_split >= 0.5).astype(int)

        n_samples = len(group_df)
        if n_samples == 0:
            continue

        metrics = {
            **group_values,
            "n_samples": float(n_samples),
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "brier_score": float(brier_score_loss(y_true, proba_split)),
            "log_loss": float(log_loss_fn(y_true, proba_split)),
            "ece": float(
                expected_calibration_error(
                    y_true, proba_split, n_bins=10, strategy="quantile"
                )
            ),
            "mean_confidence": float(max_conf_split.mean()),
            "median_confidence": float(np.median(max_conf_split)),
        }

        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true, proba_split))
        except ValueError:
            metrics["roc_auc"] = float("nan")

        for boundary in boundaries:
            metrics[f"share_below_{boundary:.2f}"] = float(
                (max_conf_split < boundary).mean()
            )

        if bootstrap_iterations:
            seed = None if random_state is None else random_state + group_idx * 4
            acc_ci = bootstrap_confidence_interval(
                accuracy_score,
                y_true,
                y_pred,
                n_bootstraps=bootstrap_iterations,
                confidence=bootstrap_confidence,
                random_state=seed,
            )
            metrics["accuracy_ci_lower"], metrics["accuracy_ci_upper"] = acc_ci

            brier_ci = bootstrap_confidence_interval(
                brier_score_loss,
                y_true,
                proba_split,
                n_bootstraps=bootstrap_iterations,
                confidence=bootstrap_confidence,
                random_state=None if seed is None else seed + 1,
            )
            metrics["brier_ci_lower"], metrics["brier_ci_upper"] = brier_ci

            log_ci = bootstrap_confidence_interval(
                log_loss_fn,
                y_true,
                proba_split,
                n_bootstraps=bootstrap_iterations,
                confidence=bootstrap_confidence,
                random_state=None if seed is None else seed + 2,
            )
            metrics["logloss_ci_lower"], metrics["logloss_ci_upper"] = log_ci

            try:
                roc_ci = bootstrap_confidence_interval(
                    roc_auc_score,
                    y_true,
                    proba_split,
                    n_bootstraps=bootstrap_iterations,
                    confidence=bootstrap_confidence,
                    random_state=None if seed is None else seed + 3,
                )
                metrics["roc_auc_ci_lower"], metrics["roc_auc_ci_upper"] = roc_ci
            except ValueError:
                metrics["roc_auc_ci_lower"] = float("nan")
                metrics["roc_auc_ci_upper"] = float("nan")

        metrics_rows.append(metrics)

        calib = build_calibration_table(
            y_true,
            proba_split,
            n_bins=10,
            strategy="quantile",
        )
        for key, value in group_values.items():
            calib[key] = value
        calibration_rows.append(calib)

        if boundaries:
            counts = (
                group_df["confidence_band"]
                .value_counts(sort=False)
                .rename_axis("confidence_band")
                .to_frame("count")
                .reset_index()
            )
            for key, value in group_values.items():
                counts[key] = value
            counts["fraction"] = counts["count"] / n_samples
            confidence_rows.append(counts)

    metrics_df = pd.DataFrame(metrics_rows)
    calibration_df = (
        pd.concat(calibration_rows, ignore_index=True)
        if calibration_rows
        else pd.DataFrame(
            columns=[
                "bin",
                "bin_lower",
                "bin_upper",
                "count",
                "mean_predicted",
                "fraction_of_positives",
                *(group_cols or ["group"]),
            ]
        )
    )
    confidence_df = (
        pd.concat(confidence_rows, ignore_index=True)
        if confidence_rows
        else pd.DataFrame(
            columns=["confidence_band", "count", "fraction", *(group_cols or ["group"])],
        )
    )

    out_path: Path | None = None
    if out_dir is not None:
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        metrics_df.to_csv(out_path / "pairwise_uncertainty_metrics.csv", index=False)
        calibration_df.to_csv(
            out_path / "pairwise_uncertainty_calibration.csv", index=False
        )
        confidence_df.to_csv(
            out_path / "pairwise_uncertainty_confidence_bands.csv", index=False
        )

        threshold = (
            uncertain_threshold
            if uncertain_threshold is not None
            else (boundaries[0] if boundaries else None)
        )
        if threshold is not None:
            low_conf = data[data["max_confidence"] < threshold]
            if not low_conf.empty:
                cols = [proba_col, "max_confidence", target_col]
                cols.extend(c for c in ["pair_id", "ind_a", "ind_b"] if c in low_conf.columns)
                cols.extend(group_cols)
                low_conf.loc[:, cols].to_csv(
                    out_path / "pairwise_uncertainty_low_confidence.csv",
                    index=False,
                )

    return {
        "metrics": metrics_df,
        "calibration": calibration_df,
        "confidence_bands": confidence_df,
        "annotated_predictions": data,
    }
