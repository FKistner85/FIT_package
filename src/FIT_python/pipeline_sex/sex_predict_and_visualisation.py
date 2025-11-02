from __future__ import annotations

import warnings
from pathlib import Path
from typing import Sequence

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    log_loss,
    roc_auc_score,
)

from FIT_python.config import DATA_DIR, PATHS, SEX_PREDICT_METRIC, CONFIG
from FIT_python.analysis.uncertainty import (
    assign_confidence_bands,
    bootstrap_confidence_interval,
    build_calibration_table,
    expected_calibration_error,
)
from FIT_python.utils import get_species_paths
from FIT_python.Visualisations.plot_style import apply_style, map_sex


# =============================================================================
# Constants / Mappings
# =============================================================================
SEX_TO_INT = {"f": 0, "m": 1}
INT_TO_SEX = {0: "f", 1: "m"}
SEX_VALUE_MAP = CONFIG["visualisation"]["sex"]["value_map"]


# =============================================================================
# Small I/O helper
# =============================================================================
def _load_split(path: Path) -> pd.DataFrame:
    """Load a split parquet file if it exists, else empty DataFrame."""
    if Path(path).exists():
        return pd.read_parquet(path)
    return pd.DataFrame()


# =============================================================================
# Baseline prediction (simple model)
# =============================================================================
def predict_simple_baseline(
    species: str,
    exp_dir: Path | None = None,
    models_dir: Path | None = None,
    *,
    include_inference: bool = True,
    reuse_csv: bool = True,
) -> pd.DataFrame:
    """Return predictions of the simple baseline model for ``species``.

    Parameters
    ----------
    species:
        Species folder under ``data/splits``.
    exp_dir:
        Optional experiment directory containing a ``models`` subdirectory.
        When provided, models are loaded from ``exp_dir / 'models'``.
    models_dir:
        Optional directory containing the trained baseline model. When both
        ``exp_dir`` and ``models_dir`` are ``None`` the path from
        :func:`get_species_paths` is used.
    include_inference:
        Include the ``inference`` split when it exists.
    reuse_csv:
        When ``True`` the function expects ``{species}_baseline_predictions.csv``
        to exist in the predictions directory. If the file is missing a
        ``FileNotFoundError`` is raised. Set to ``False`` to recompute the
        predictions.

    Returns
    -------
    pandas.DataFrame
        DataFrame with the original splits and three additional columns:
        ``pred_baseline_sex``, ``pred_baseline_proba_f`` and
        ``pred_baseline_proba_m``.
    """
    paths = get_species_paths(section="sex_modelling", species=species)
    splits_dir = PATHS["splits"] / species  # global splits path (authoritative)

    # Where to write/read prediction CSVs
    pred_dir = (
        Path(paths["predictions"])
        if "predictions" in paths
        else (PATHS["sex_modelling"] / species / "tables")
    )
    pred_dir.mkdir(parents=True, exist_ok=True)
    csv_path = pred_dir / f"{species}_baseline_predictions.csv"

    # Where to load the model from
    if models_dir is not None:
        model_dir = Path(models_dir)
    elif exp_dir is not None:
        model_dir = Path(exp_dir) / "models"
    else:
        model_dir = Path(paths["models"])
    model_path = model_dir / f"{species}.joblib"

    if reuse_csv:
        if csv_path.exists():
            return pd.read_csv(csv_path)
        raise FileNotFoundError(
            f"Predictions CSV not found: {csv_path}. "
            "Set reuse_csv=False to recompute predictions."
        )

    if not splits_dir.is_dir():
        warnings.warn(
            f"Split directory for {species!r} not found – skipping.",
            UserWarning,
        )
        return pd.DataFrame()

    clf = joblib.load(model_path)

    split_names = ["train", "test"]
    if include_inference and (splits_dir / "inference.parquet").exists():
        split_names.append("inference")

    dfs: list[pd.DataFrame] = []
    for name in split_names:
        df = _load_split(splits_dir / f"{name}.parquet")
        if df.empty:
            continue

        drop_cols = ["sex"]
        if "individual_id" in df.columns:
            drop_cols.append("individual_id")
        X = df.drop(columns=[c for c in drop_cols if c in df.columns])
        X = X.select_dtypes(include=["number"]).copy()
        if "Fold" in X.columns:
            X = X.drop(columns=["Fold"])

        if X.empty:
            continue

        df["__split__"] = name
        df["pred_baseline_sex"] = clf.predict(X)
        proba = clf.predict_proba(X)
        df["pred_baseline_proba_f"] = proba[:, 0]
        df["pred_baseline_proba_m"] = proba[:, 1]
        dfs.append(df)

    all_df = pd.concat(dfs, ignore_index=True)

    if not reuse_csv:
        all_df.to_csv(csv_path, index=False)

    return all_df


# =============================================================================
# Best-model predictions (per species)
# =============================================================================
def predict_all(
    species: str,
    metric_key: str = SEX_PREDICT_METRIC,  # e.g. "balanced_accuracy"
    prefer_generic: bool = False,          # kept for API compatibility
    models_dir: str | Path | None = None,
    include_inference: bool = True,
    reuse_csv: bool = False,
    use_cv_train_predictions: bool = True,
) -> pd.DataFrame:
    """Return dataframe with best-model predictions for Train/Test/Inference.

    Train: Out-of-fold predictions (OOF) using 'Fold' column.
    Test + Inference: Predictions from model retrained on full Train.
    """
    paths = get_species_paths(section="sex_modelling", species=species)
    splits_dir = PATHS["splits"] / species  # global splits path (authoritative)

    # models_dir either provided or resolved from paths
    models_dir = Path(models_dir) if models_dir is not None else Path(paths["models"])

    # Where to write/read prediction CSVs
    pred_dir = (
        Path(paths["predictions"])
        if "predictions" in paths
        else (PATHS["sex_modelling"] / species / "tables")
    )
    pred_dir.mkdir(parents=True, exist_ok=True)
    csv_path = pred_dir / f"{species}_all_predictions.csv"

    # When reuse_csv=True the predictions must already exist on disk
    if reuse_csv:
        if csv_path.exists():
            return pd.read_csv(csv_path)
        raise FileNotFoundError(
            f"Predictions CSV not found: {csv_path}. "
            "Set reuse_csv=False to recompute predictions."
        )

    if not splits_dir.is_dir():
        warnings.warn(f"Split directory for {species!r} not found – skipping.")
        return pd.DataFrame()

    # Load splits
    split_names = ["train", "test"]
    if include_inference and (splits_dir / "inference.parquet").exists():
        split_names.append("inference")
    splits = {n: splits_dir / f"{n}.parquet" for n in split_names}
    dfs = {name: pd.read_parquet(p) for name, p in splits.items()}

    for name, df in dfs.items():
        df.columns = (
            df.columns.str.replace(r"[.\-]", "_", regex=True).str.replace("T", "t")
        )
        df["__split__"] = name

    # Load best model
    best_model_path = models_dir / metric_key / f"{species}.joblib"
    if not best_model_path.exists():
        raise FileNotFoundError(f"No model found at {best_model_path}")
    best_model = joblib.load(best_model_path)

    # Features (numeric, excluding prediction columns and 'Fold')
    num_cols = dfs["train"].select_dtypes(include=np.number).columns
    feature_cols = [c for c in num_cols if not c.startswith("pred_") and c != "Fold"]

    # OOF predictions for Train
    if use_cv_train_predictions and "Fold" in dfs["train"].columns:
        n_samples = len(dfs["train"])
        oof_preds = np.empty(n_samples, dtype=object)
        oof_proba_f = np.empty(n_samples, dtype=float)
        oof_proba_m = np.empty(n_samples, dtype=float)

        for fold in sorted(dfs["train"]["Fold"].unique()):
            tr_idx = dfs["train"].index[dfs["train"]["Fold"] != fold]
            val_idx = dfs["train"].index[dfs["train"]["Fold"] == fold]

            X_tr = dfs["train"].loc[tr_idx, feature_cols]
            y_tr = dfs["train"].loc[tr_idx, "sex"].map(SEX_TO_INT)
            X_val = dfs["train"].loc[val_idx, feature_cols]

            mdl = clone(best_model).fit(X_tr, y_tr)
            preds = mdl.predict(X_val)
            proba = mdl.predict_proba(X_val)

            oof_preds[val_idx] = np.vectorize(INT_TO_SEX.get)(preds)
            oof_proba_f[val_idx] = proba[:, 0]
            oof_proba_m[val_idx] = proba[:, 1]

        dfs["train"]["pred_sex"] = oof_preds
        dfs["train"]["pred_proba_f"] = oof_proba_f
        dfs["train"]["pred_proba_m"] = oof_proba_m
    else:
        X_train = dfs["train"][feature_cols]
        y_train = dfs["train"]["sex"].map(SEX_TO_INT)
        mdl = clone(best_model).fit(X_train, y_train)
        proba = mdl.predict_proba(X_train)
        train_preds = mdl.predict(X_train)
        dfs["train"]["pred_sex"] = np.vectorize(INT_TO_SEX.get)(train_preds)
        dfs["train"]["pred_proba_f"] = proba[:, 0]
        dfs["train"]["pred_proba_m"] = proba[:, 1]

    # Retrain on full train for Test and Inference
    X_train_full = dfs["train"][feature_cols]
    y_train_full = dfs["train"]["sex"].map(SEX_TO_INT)
    final_model = clone(best_model).fit(X_train_full, y_train_full)

    for split in ["test", "inference"]:
        if split in dfs:
            X_split = dfs[split][feature_cols]
            proba = final_model.predict_proba(X_split)
            preds = final_model.predict(X_split)
            dfs[split]["pred_sex"] = np.vectorize(INT_TO_SEX.get)(preds)
            dfs[split]["pred_proba_f"] = proba[:, 0]
            dfs[split]["pred_proba_m"] = proba[:, 1]

    # Merge and save
    all_df = pd.concat(dfs.values(), ignore_index=True)
    all_df.to_csv(csv_path, index=False)
    return all_df


# =============================================================================
# Uncertainty & calibration summary
# =============================================================================
def summarise_uncertainty(
    df: pd.DataFrame,
    *,
    species: str | None = None,
    positive_label: str = "m",
    n_bins: int = 10,
    strategy: str = "quantile",
    confidence_boundaries: Sequence[float] = (0.6, 0.8, 0.9),
    bootstrap_iterations: int = 500,
    bootstrap_confidence: float = 0.95,
    random_state: int | None = None,
    out_dir: Path | str | None = None,
    create_plot: bool = True,
) -> dict[str, pd.DataFrame]:
    """Compute calibration and confidence metrics for a sex prediction dataframe.

    Parameters
    ----------
    df:
        Dataframe returned by :func:`predict_all` containing prediction
        probabilities and split information.
    species:
        Optional species name used in plot titles and captions.
    positive_label:
        Target label treated as the positive class. Defaults to ``"m"``.
    n_bins, strategy:
        Parameters forwarded to :func:`build_calibration_table` and
        :func:`expected_calibration_error`.
    confidence_boundaries:
        Boundaries for :func:`assign_confidence_bands`. When empty no discrete
        confidence bands are created.
    bootstrap_iterations, bootstrap_confidence, random_state:
        Configuration for :func:`bootstrap_confidence_interval`. Set
        ``bootstrap_iterations`` to ``0`` to disable interval estimation.
    out_dir:
        Optional directory for writing CSV summaries and plots.
    create_plot:
        When ``True`` a reliability diagram is written to ``out_dir``.

    Returns
    -------
    dict[str, pandas.DataFrame]
        Dictionary containing dataframes for metrics, calibration bins,
        confidence band counts and the annotated predictions.
    """

    if "__split__" not in df.columns:
        raise KeyError("DataFrame requires a '__split__' column for grouping")
    if "sex" not in df.columns:
        raise KeyError("DataFrame is missing the 'sex' column")

    positive_label = str(positive_label).lower()
    if positive_label not in {"m", "f"}:
        raise ValueError("positive_label must be 'm' or 'f'")

    proba_col = f"pred_proba_{positive_label}"
    if proba_col not in df.columns:
        raise KeyError(f"Prediction column '{proba_col}' not found")

    other_label = "f" if positive_label == "m" else "m"
    other_proba_col = f"pred_proba_{other_label}"
    df = df.copy()

    df["sex"] = df["sex"].astype(str).str.lower()
    if "pred_sex" in df.columns:
        df["pred_sex"] = df["pred_sex"].astype(str).str.lower()
    else:
        df["pred_sex"] = np.where(df[proba_col] >= 0.5, positive_label, other_label)

    proba = df[proba_col].astype(float)
    if other_proba_col in df.columns:
        other_proba = df[other_proba_col].astype(float)
        max_conf = np.maximum(proba, other_proba)
    else:
        max_conf = np.maximum(proba, 1 - proba)
    df["max_confidence"] = max_conf

    boundaries = tuple(confidence_boundaries) if confidence_boundaries else tuple()
    if boundaries:
        df["confidence_band"] = assign_confidence_bands(
            df["max_confidence"], boundaries=boundaries
        )

    metrics_rows: list[dict[str, float]] = []
    calibration_rows: list[pd.DataFrame] = []
    confidence_rows: list[pd.DataFrame] = []

    log_loss_fn = lambda y, p: log_loss(
        y,
        np.clip(p, 1e-12, 1 - 1e-12),
        labels=[0, 1],
    )

    groups = df.groupby("__split__", sort=False)
    for split_idx, (split_name, split_df) in enumerate(groups):
        y_true = (split_df["sex"] == positive_label).astype(int).to_numpy()
        y_pred = (split_df["pred_sex"] == positive_label).astype(int).to_numpy()
        proba_split = split_df[proba_col].astype(float).to_numpy()
        max_conf_split = split_df["max_confidence"].to_numpy()

        n_samples = len(split_df)
        if n_samples == 0:
            continue

        metrics: dict[str, float] = {
            "split": split_name,
            "n_samples": float(n_samples),
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "brier_score": float(brier_score_loss(y_true, proba_split)),
            "log_loss": float(log_loss_fn(y_true, proba_split)),
            "ece": float(
                expected_calibration_error(
                    y_true,
                    proba_split,
                    n_bins=n_bins,
                    strategy=strategy,
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
            seed = None if random_state is None else random_state + split_idx
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
            n_bins=n_bins,
            strategy=strategy,
        ).assign(split=split_name)
        calibration_rows.append(calib)

        if boundaries:
            counts = (
                split_df["confidence_band"]
                .value_counts(sort=False)
                .rename_axis("confidence_band")
                .to_frame("count")
                .reset_index()
            )
            counts["split"] = split_name
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
                "split",
            ]
        )
    )
    confidence_df = (
        pd.concat(confidence_rows, ignore_index=True)
        if confidence_rows
        else pd.DataFrame(columns=["confidence_band", "count", "split", "fraction"])
    )

    out_path: Path | None = None
    if out_dir is not None:
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        metrics_df.to_csv(out_path / "sex_uncertainty_metrics.csv", index=False)
        calibration_df.to_csv(out_path / "sex_uncertainty_calibration.csv", index=False)
        confidence_df.to_csv(
            out_path / "sex_uncertainty_confidence_bands.csv", index=False
        )

        if boundaries:
            first_boundary = boundaries[0]
            low_conf = df[df["max_confidence"] < first_boundary]
            if not low_conf.empty:
                cols = ["__split__", "sex", "pred_sex", proba_col, "max_confidence"]
                if other_proba_col in df.columns:
                    cols.append(other_proba_col)
                for opt in ["individual_id", "Trail", "trail", "sample_id"]:
                    if opt in df.columns and opt not in cols:
                        cols.append(opt)
                low_conf.loc[:, cols].to_csv(
                    out_path / "sex_uncertainty_low_confidence.csv", index=False
                )

        if create_plot and not calibration_df.empty:
            apply_style()
            fig, ax = plt.subplots(figsize=(6, 4))
            for split_name, split_df in calibration_df.groupby("split"):
                ax.plot(
                    split_df["mean_predicted"],
                    split_df["fraction_of_positives"],
                    marker="o",
                    label=str(split_name),
                )
            ax.plot([0, 1], [0, 1], linestyle="--", color="black", linewidth=1)
            title = "Sex model reliability"
            if species:
                title += f" – {species}"
            ax.set_title(title)
            ax.set_xlabel("Mean predicted probability")
            ax.set_ylabel("Fraction of positives")
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.legend(title="Split")
            fig.tight_layout()
            plot_path = out_path / "sex_uncertainty_reliability.png"
            fig.savefig(plot_path)
            try:
                from FIT_python.caption_utils import save_caption

                save_caption(
                    plot_path,
                    "Reliability diagram comparing predicted and empirical sex probabilities.",
                )
            except Exception:
                pass
            plt.close(fig)

    return {
        "metrics": metrics_df,
        "calibration": calibration_df,
        "confidence_bands": confidence_df,
        "annotated_predictions": df,
    }


def create_uncertainty_report(
    species: str,
    *,
    metric_key: str = SEX_PREDICT_METRIC,
    models_dir: str | Path | None = None,
    include_inference: bool = True,
    reuse_csv: bool = False,
    use_cv_train_predictions: bool = True,
    out_dir: str | Path | None = None,
    positive_label: str = "m",
    n_bins: int = 10,
    strategy: str = "quantile",
    confidence_boundaries: Sequence[float] = (0.6, 0.8, 0.9),
    bootstrap_iterations: int = 500,
    bootstrap_confidence: float = 0.95,
    random_state: int | None = None,
    create_plot: bool = True,
) -> dict[str, pd.DataFrame]:
    """Generate confidence and calibration artefacts for a species."""

    df = predict_all(
        species,
        metric_key=metric_key,
        prefer_generic=False,
        models_dir=models_dir,
        include_inference=include_inference,
        reuse_csv=reuse_csv,
        use_cv_train_predictions=use_cv_train_predictions,
    )

    if df.empty:
        return {
            "metrics": pd.DataFrame(),
            "calibration": pd.DataFrame(),
            "confidence_bands": pd.DataFrame(),
            "annotated_predictions": df,
        }

    if out_dir is None:
        paths = get_species_paths(section="sex_modelling", species=species)
        out_dir = paths["tables"] / "uncertainty"

    return summarise_uncertainty(
        df,
        species=species,
        positive_label=positive_label,
        n_bins=n_bins,
        strategy=strategy,
        confidence_boundaries=confidence_boundaries,
        bootstrap_iterations=bootstrap_iterations,
        bootstrap_confidence=bootstrap_confidence,
        random_state=random_state,
        out_dir=out_dir,
        create_plot=create_plot,
    )


# =============================================================================
# Hyperparameter search visualisation
# =============================================================================
def plot_hyperparam_heatmap(df: pd.DataFrame, out_dir: Path) -> Path:
    """Plot a heatmap visualising mean CV accuracy across preprocessing options."""
    if df.empty:
        raise ValueError("DataFrame for heatmap is empty")

    # Accept column names from BayesSearchCV output for convenience
    col_map = {
        "select__method": "fs_method",
        "reduce_pre__method": "reduce_pre_method",
        "mean_test_score": "cv_balanced_accuracy",
        "mean_test_balanced_accuracy": "cv_balanced_accuracy",
    }
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

    # Fill missing pipeline step columns to keep the plot resilient
    if "fs_method" not in df.columns:
        df["fs_method"] = "None"
    if "reduce_pre_method" not in df.columns:
        df["reduce_pre_method"] = "None"
    if "cv_balanced_accuracy" not in df.columns:
        raise KeyError("Missing column 'cv_balanced_accuracy' for heatmap")

    pivot = (
        df.pivot_table(
            index="fs_method",
            columns="reduce_pre_method",
            values="cv_balanced_accuracy",
            aggfunc="mean",
        )
        .sort_index()
        .sort_index(axis=1)
    )

    apply_style()
    plt.figure(figsize=(6, 4))
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="viridis", vmin=0, vmax=1)
    plt.xlabel("Pre-dimensionality reducer")
    plt.ylabel("Feature selector")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "hyperparam_heatmap.png"

    from FIT_python.caption_utils import save_caption

    plt.tight_layout()
    plt.savefig(out_file)
    save_caption(out_file, "Mean CV accuracy by selector and reducer")
    plt.close()
    return out_file


# =============================================================================
# Batch prediction for all species
# =============================================================================
def predict_all_species(species_list: list[str] | None = None) -> pd.DataFrame:
    """Predict sex for all species and combine into a single CSV."""
    if species_list is None:
        species_list = [p.name for p in (DATA_DIR / "splits").iterdir() if p.is_dir()]

    valid_species: list[str] = []
    for species in species_list:
        if (DATA_DIR / "splits" / species).is_dir():
            valid_species.append(species)
        else:
            warnings.warn(
                f"Split directory for {species!r} not found – skipping.",
                UserWarning,
            )

    dfs = [predict_all(sp, prefer_generic=True, reuse_csv=False) for sp in valid_species]
    if not dfs:
        return pd.DataFrame()

    all_df = pd.concat(dfs, ignore_index=True)
    out_csv = PATHS["sex_modelling"] / "tables" / "all_species_all_predictions.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    all_df.to_csv(out_csv, index=False)
    return all_df


# =============================================================================
# Confusion matrices (train/test)
# =============================================================================
def plot_confusion(df: pd.DataFrame) -> None:
    """Plot confusion matrices for train (CV) and test predictions incl. counts."""
    apply_style()

    if "pred_sex" not in df.columns:
        raise KeyError("DataFrame contains no 'pred_sex' column")
    if "__split__" not in df.columns:
        raise KeyError("DataFrame is missing the '__split__' column")

    mapping = {"f": "F", "m": "M", "female": "F", "male": "M", "F": "F", "M": "M", 0: "F", 1: "M"}

    df_mapped = df.copy()
    df_mapped["true_label"] = df_mapped["sex"].map(mapping)
    df_mapped["pred_label"] = df_mapped["pred_sex"].map(mapping)

    train = df_mapped[
        (df_mapped["__split__"] == "train")
        & df_mapped["true_label"].isin(["F", "M"])
        & df_mapped["pred_label"].isin(["F", "M"])
    ]
    test = df_mapped[
        (df_mapped["__split__"] == "test")
        & df_mapped["true_label"].isin(["F", "M"])
        & df_mapped["pred_label"].isin(["F", "M"])
    ]
    if train.empty or test.empty:
        return

    y_true_train, y_pred_train = train["true_label"], train["pred_label"]
    y_true_test, y_pred_test = test["true_label"], test["pred_label"]

    cm_train = confusion_matrix(y_true_train, y_pred_train, labels=["F", "M"])
    cm_test = confusion_matrix(y_true_test, y_pred_test, labels=["F", "M"])

    height = plt.rcParams["figure.figsize"][1] * 0.6
    fig, axes = plt.subplots(1, 2, figsize=(8, height), sharey=True)
    mats = [(cm_train, "a)"), (cm_test, "b)")]

    for ax, (cm, title) in zip(axes, mats):
        cm_sum = cm.sum(axis=1, keepdims=True)
        cm_perc = cm / cm_sum.astype(float) * 100

        annot = np.empty_like(cm).astype(str)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                c = cm[i, j]
                p = cm_perc[i, j]
                annot[i, j] = f"{c} ({p:.0f}%)"

        sns.heatmap(
            cm_perc,
            annot=annot,
            fmt="",
            cmap="Blues",
            xticklabels=["Female", "Male"],
            yticklabels=["Female", "Male"],
            ax=ax,
        )
        ax.set_title(title, loc="left", fontweight="bold")
        ax.set_xlabel("Predicted Sex")
        if ax is axes[0]:
            ax.set_ylabel("True Sex")
        else:
            ax.set_ylabel("")
            ax.tick_params(axis="y", labelleft=False)
        ax.set_xticklabels(["Female", "Male"], rotation=0)
        ax.set_yticklabels(["Female", "Male"], rotation=0)

    plt.tight_layout()
    plt.show()


# =============================================================================
# Helper for majority/pivot sorting (kept for compatibility)
# =============================================================================
def build_pivot_sorted(sub: pd.DataFrame, col: str) -> tuple[pd.DataFrame, int]:
    tmp = sub.copy()
    tmp[col] = tmp[col].map(SEX_VALUE_MAP)

    # True sex (True = Female, False = Male)
    if "sex" not in tmp.columns:
        raise KeyError("DataFrame lacks 'sex' column for true sex sorting")
    tmp["true_female"] = tmp["sex"].map(SEX_VALUE_MAP) == "Female"

    # Pivot (absolute counts)
    pivot = tmp.pivot_table(index="individual_id", columns=col, aggfunc="size", fill_value=0)
    pivot = pivot.reindex(columns=["Female", "Male"], fill_value=0)

    # Ratios for sorting
    total_counts = pivot.sum(axis=1)
    female_ratio = pivot["Female"] / total_counts
    male_ratio = pivot["Male"] / total_counts

    # IDs per true sex group
    true_female_ids = tmp.loc[tmp["true_female"], "individual_id"].unique()
    true_male_ids = tmp.loc[~tmp["true_female"], "individual_id"].unique()

    # Symmetric sorting
    female_df = (
        pivot.loc[pivot.index.isin(true_female_ids)]
        .assign(ratio=female_ratio)
        .sort_values(by="ratio", ascending=False)
        .drop(columns="ratio")
    )
    male_df = (
        pivot.loc[pivot.index.isin(true_male_ids)]
        .assign(ratio=male_ratio)
        .sort_values(by="ratio", ascending=True)
        .drop(columns="ratio")
    )

    pivot_sorted = pd.concat([female_df, male_df])
    female_count = len(female_df)
    return pivot_sorted, female_count


def plot_hist_predsex_train_test(df: pd.DataFrame) -> None:
    """Placeholder kept for compatibility (no-op if unused)."""
    apply_style()
    pred_cols = [c for c in df if c.startswith("pred_") and c.endswith("_sex")]
    if not pred_cols or "__split__" not in df.columns:
        return
    # Intentionally left minimal to stay functionally neutral.


# =============================================================================
# Inference bar plot (per location/trail)
# =============================================================================
def plot_inference(df: pd.DataFrame) -> None:
    """Plot predicted sex counts for inference split with Location + Trail number labels."""
    apply_style()
    pred_cols = [c for c in df if c.startswith("pred_") and c.endswith("_sex")]
    if not pred_cols:
        raise KeyError("DataFrame contains no prediction columns")
    if "inference" not in df["__split__"].unique():
        return

    sub = df[df["__split__"] == "inference"].copy()
    for col in pred_cols:
        sub[col] = sub[col].map(CONFIG["visualisation"]["sex"]["value_map"])

        # Extract trailing number from 'trail'
        sub["trail_num"] = sub["trail"].str.extract(r"(\d+)$")
        sub["loc_trail"] = sub["location"] + " " + sub["trail_num"].fillna("")

        pivot = sub.pivot_table(index="loc_trail", columns=col, aggfunc="size", fill_value=0)

        colors = [CONFIG["visualisation"]["sex"]["colors"].get(c, "#333333") for c in pivot.columns]
        ax = pivot.plot.bar(stacked=True, figsize=(10, 4), color=colors)

        plt.xlabel("Location & Trail")
        plt.ylabel("Count")
        ax.legend(title="Predicted", labels=pivot.columns)
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.show()


def plot_confusion_and_inference(df: pd.DataFrame) -> None:
    """Backward compatible wrapper calling :func:`plot_confusion` and :func:`plot_inference`."""
    plot_confusion(df)
    plot_inference(df)


# =============================================================================
# Quality heatmaps
# =============================================================================
def plot_quality_grouped(df: pd.DataFrame) -> None:
    """Plot quality heatmaps based on majority decision per Trail or Individual."""
    apply_style()

    splits = ["train", "test"] if "__split__" in df.columns else [None]
    for split in splits:
        df_sub = df if split is None else df[df["__split__"] == split]
        if df_sub.empty:
            continue

        # Only f/m
        df_plot = df_sub[df_sub["sex"].isin(["f", "m"])].copy()
        df_plot["true_label"] = map_sex(df_plot["sex"])

        # Prediction columns
        pred_cols = [c for c in df_plot if c.startswith("pred_") and c.endswith("_sex")]
        if not pred_cols:
            raise KeyError("No prediction label columns found in dataframe")

        # Per-print predicted label columns for majority logic
        for col in pred_cols:
            key = col.split("_")[1]
            df_plot[f"pred_label_{key}"] = df_plot[col].map({0: "Female", 1: "Male"})

        # Majority classifier per group
        def classify_majority(group: pd.DataFrame) -> str:
            correct_ratio = (group[pred_label_cols].eq(group["true_label"], axis=0).any(axis=1)).mean()
            if correct_ratio >= 0.9:
                return "High"
            elif correct_ratio >= 0.7:
                return "Moderate"
            elif correct_ratio >= 0.5:
                return "Low"
            else:
                return "Misclassified"

        pred_label_cols = [c for c in df_plot if c.startswith("pred_label_")]

        cmap = LinearSegmentedColormap.from_list("green", ["white", "mediumseagreen"])
        height = plt.rcParams["figure.figsize"][1] * 0.75
        fig, axes = plt.subplots(
            1,
            2,
            figsize=(plt.rcParams["figure.figsize"][0], height),
            sharey=True,
        )

        for ax, (tag, cols) in zip(
            axes,
            [
                ("c)", ["trail", "true_label"]),
                ("d)", ["individual_id", "true_label"]),
            ],
        ):
            acc = df_plot.groupby(cols).apply(classify_majority).reset_index(name="Class")

            pivot = (
                acc.pivot_table(
                    index="true_label",
                    columns="Class",
                    values=cols[0],
                    aggfunc="count",
                    fill_value=0,
                )
                .reindex(columns=["High", "Moderate", "Low", "Misclassified"], fill_value=0)
            )

            proportions = pivot.div(pivot.sum(axis=1), axis=0).fillna(0)
            counts = pivot.astype(int)
            total = counts.values.sum()

            # Annotation: counts + global percentage
            annot = counts.copy().astype(str)
            for i in counts.index:
                for j in counts.columns:
                    v = counts.at[i, j]
                    annot.at[i, j] = f"{v} ({v/total:.0%})" if v > 0 else ""

            sns.heatmap(
                proportions,
                annot=annot,
                fmt="",
                cmap=cmap,
                vmin=0,
                vmax=1,
                linewidths=0.5,
                linecolor="gray",
                ax=ax,
            )
            ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
            ax.set_yticklabels(["Female", "Male"], rotation=0)
            ax.set_title(tag, loc="left", fontweight="bold")
            ax.set_xlabel("Prediction Confidence")
            if ax is axes[0]:
                ax.set_ylabel("Sex")
            else:
                ax.set_ylabel("")
                ax.tick_params(axis="y", labelleft=False)

        plt.tight_layout()
        plt.show()


def _plot_quality_heatmaps_single(
    df_sub: pd.DataFrame,
    pred_col: str,
    proba_cols: list[str],
    title: str | None = None,
    group_by: str | None = None,
) -> None:
    """Plot prediction quality heatmaps for a single model and split."""
    df = df_sub.copy()
    df["pred_label"] = df[pred_col].map({0: "Female", 1: "Male"})
    df["Correct"] = df["pred_label"] == df["true_label"]
    df["Max_Prob"] = df[proba_cols].max(axis=1)
    df["Quality"] = df["Max_Prob"].apply(lambda p: "High" if p > 0.9 else ("Moderate" if p > 0.7 else "Low"))

    if group_by is not None:
        grouped = df.groupby([group_by, "true_label"])
        df = grouped.agg(correct_rate=("Correct", "mean")).reset_index()
        df["Correct"] = df["correct_rate"] > 0.5
        df["Quality"] = df["correct_rate"].apply(
            lambda p: (
                "High"
                if p > 0.9
                else ("Moderate" if p > 0.7 else ("Low" if p > 0.5 else "Misclassified"))
            )
        )

    idx = pd.MultiIndex.from_product([["Female", "Male"], [True, False]], names=["true_label", "Correct"])
    columns = ["High", "Moderate", "Low"]
    if group_by is not None:
        columns.append("Misclassified")
    counts = (
        df.groupby(["true_label", "Correct", "Quality"]).size().unstack(fill_value=0)
    ).reindex(index=idx, columns=columns, fill_value=0)

    if counts.values.sum() == 0:
        print("    → No data to plot")
        return

    normed = counts.div(counts.sum(axis=1), axis=0).fillna(0)

    def make_annot(block: pd.DataFrame) -> pd.DataFrame:
        total = block.values.sum()
        return block.applymap(lambda x: (f"{int(x)}\n({int(round(x / total * 100))}%)" if total > 0 else "0\n(0%)"))

    annot_corr = make_annot(counts.xs(True, level="Correct"))
    annot_incorr = make_annot(counts.xs(False, level="Correct"))

    green_cmap = LinearSegmentedColormap.from_list("green", ["white", "mediumseagreen"])
    red_cmap = LinearSegmentedColormap.from_list("red", ["white", "crimson"])

    height = plt.rcParams["figure.figsize"][1] * 0.75
    fig, axes = plt.subplots(1, 2, figsize=(plt.rcParams["figure.figsize"][0], height), sharey=True)
    sns.heatmap(
        normed.xs(True, level="Correct"),
        annot=annot_corr,
        fmt="",
        cmap=green_cmap,
        vmin=0,
        vmax=1,
        linewidths=0.5,
        linecolor="gray",
        ax=axes[0],
        cbar=True,
    )
    axes[0].set_title("a)", loc="left", fontweight="bold")
    axes[0].set(ylabel="Sex")
    sns.heatmap(
        normed.xs(False, level="Correct"),
        annot=annot_incorr,
        fmt="",
        cmap=red_cmap,
        vmin=0,
        vmax=1,
        linewidths=0.5,
        linecolor="gray",
        ax=axes[1],
        cbar=True,
    )
    axes[1].set_title("b)", loc="left", fontweight="bold")
    axes[1].set(ylabel="")
    for ax in axes:
        ax.set_xlabel("Prediction Confidence")
        labels = ["High", "Moderate", "Low"]
        if group_by is not None:
            labels.append("Misclassified")
        ax.set_xticklabels(labels, rotation=0)
        ax.set_yticklabels(["Female", "Male"], rotation=0)
    plt.tight_layout()
    plt.show()


def plot_quality_heatmaps(
    df: pd.DataFrame,
    pred_col: str | None = None,
    proba_cols: list[str] | None = None,
    title: str | None = None,
    group_by: str | None = None,
) -> None:
    """Plot prediction-quality heatmaps."""
    apply_style()

    if pred_col is not None and proba_cols is not None:
        splits = ["train", "test"] if "__split__" in df.columns else [None]
        for split in splits:
            df_sub = df if split is None else df[df["__split__"] == split]
            if df_sub.empty:
                continue
            df_sub = df_sub.copy()
            df_sub["true_label"] = map_sex(df_sub["sex"])
            sub_title = title if title is not None else (f"{split} set" if split else None)
            _plot_quality_heatmaps_single(df_sub, pred_col, proba_cols, sub_title, group_by)
        return

    # Automatic generation for all models and splits
    df = df.copy()
    df["true_label"] = map_sex(df["sex"])

    pred_cols = [c for c in df if c.startswith("pred_") and c.endswith("_sex")]
    if not pred_cols:
        raise KeyError("DataFrame contains no prediction columns")

    split_order = [s for s in ["train", "test"] if s in df["__split__"].unique()]
    for split in split_order:
        for col in pred_cols:
            model = col[len("pred_") : -len("_sex")]
            model_prefix = f"{model}_" if model else ""
            probs = [f"pred_{model_prefix}proba_f", f"pred_{model_prefix}proba_m"]
            df_sub = df[df["__split__"] == split].copy()
            df_sub = df_sub[df_sub["sex"].isin(["f", "m"])]
            df_sub = df_sub[df_sub[col].isin([0, 1])]
            if df_sub.empty:
                continue
            _plot_quality_heatmaps_single(
                df_sub,
                col,
                probs,
                title=f"{model} — {split}",
                group_by=group_by,
            )


def plot_model_quality_heatmaps(df: pd.DataFrame) -> None:
    """Deprecated wrapper for :func:`plot_quality_heatmaps`."""
    warnings.warn(
        "plot_model_quality_heatmaps is deprecated; use plot_quality_heatmaps",
        DeprecationWarning,
        stacklevel=2,
    )
    plot_quality_heatmaps(df)


# =============================================================================
# Individual probability histograms
# =============================================================================
def plot_individual_probabilities(df: pd.DataFrame, out_dir: str | Path) -> None:
    """Plot distribution of predicted sex probabilities for train and test."""
    apply_style()
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # We detect any columns ending with '_proba_m' (backwards compatible)
    proba_cols = [c for c in df.columns if c.endswith("_proba_m")]
    if not proba_cols:
        raise ValueError("DataFrame contains no probability columns ending with '_proba_m'.")

    palette = {
        "Female": CONFIG["visualisation"]["sex"]["colors"]["Female"],
        "Male": CONFIG["visualisation"]["sex"]["colors"]["Male"],
        "Unknown": CONFIG["visualisation"]["sex"]["colors"].get("Unknown", "#333333"),
    }

    splits = ["train", "test"] if "__split__" in df.columns else [None]
    for split in splits:
        df_sub = df if split is None else df[df["__split__"] == split]
        if df_sub.empty:
            continue
        agg = {c: "mean" for c in proba_cols}
        agg["sex"] = "first"
        grouped = df_sub.groupby("individual_id").agg(agg).reset_index()
        grouped["sex_std"] = map_sex(grouped["sex"])

        for col in proba_cols:
            # Derive a readable model identifier from column name.
            model = col[len("pred_") : -len("_proba_m")] or "pred"
            height = plt.rcParams["figure.figsize"][1] * 0.75
            plt.figure(figsize=(plt.rcParams["figure.figsize"][0], height))
            sns.histplot(
                grouped,
                x=col,
                hue="sex_std",
                element="step",
                stat="density",
                common_norm=False,
                palette=palette,
            )
            plt.xticks(rotation=0)
            plt.xlabel("Predicted probability male")
            plt.ylabel("Density")

            from FIT_python.caption_utils import save_caption

            plt.tight_layout()
            suffix = f"_{split}" if split is not None else ""
            file = out_path / f"{model}_individual_probabilities{suffix}.png"
            plt.savefig(file)
            save_caption(file, f"Predicted male probability for {model}{suffix.replace('_', ' ')}".strip())
            plt.close()