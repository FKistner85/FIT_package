from __future__ import annotations

"""Lightweight baseline sex classifier."""

from pathlib import Path
from typing import List, Dict, Any

import pandas as pd
from joblib import dump, load

from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_val_score, PredefinedSplit
from tqdm.auto import tqdm
from tqdm_joblib import tqdm_joblib
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_recall_fscore_support,
)

from FIT_python.config import SPLITS_DIR, DATA_DIR
from FIT_python.general_pipeline_steps.feature_selection_wrapper import (
    FeatureSelectionTransformer,
)
from . import grouped_metrics


def _load_split(fp: Path) -> pd.DataFrame:
    df = pd.read_parquet(fp)
    df = df.dropna(subset=["sex"])
    df = df.query("sex in ['f','m']").reset_index(drop=True)
    return df


def run_simple_baseline_all_species(
    exp_dir: Path,
    n_jobs: int = -1,
    progress: bool = False,
    max_features: int = 10,
    reuse_results: bool = True,
    save_predictions: bool = True,
) -> pd.DataFrame:
    """Train a simple LDA baseline for each species.

    Parameters
    ----------
    exp_dir:
        Directory where ``raw_results.csv`` and models will be stored.

    n_jobs:
        Number of CPU cores to use during cross-validation. ``-1``
        uses all available cores.

    progress:
        Show a progress bar for species and cross-validation when ``True``.

    max_features:
        Maximum number of features to select during forward selection.

    reuse_results:
        When ``True`` existing ``raw_results.csv`` and model files in
        ``exp_dir`` are loaded and returned instead of training new models.

    save_predictions:
        When ``True`` baseline predictions for each species are written to
        ``{exp_dir}/{species}_baseline_predictions.csv`` after training.

    Returns
    -------
    pandas.DataFrame
        Table with one row per species containing the evaluation metrics.
    """
    exp_dir = Path(exp_dir)

    if reuse_results:
        csv_path = exp_dir / "raw_results.csv"
        model_dir = exp_dir / "models"
        if csv_path.exists() and model_dir.is_dir():
            df = pd.read_csv(csv_path)
            if not df.empty and all((model_dir / f"{sp}.joblib").exists() for sp in df["species"]):
                return df

    exp_dir.mkdir(parents=True, exist_ok=True)
    model_dir = exp_dir / "models"
    model_dir.mkdir(exist_ok=True)

    species_dirs = [d for d in sorted(SPLITS_DIR.iterdir()) if d.is_dir()]
    if not species_dirs:
        raise FileNotFoundError(f"No split directories found in {SPLITS_DIR}")

    iter_dirs = tqdm(species_dirs, desc="Species") if progress else species_dirs

    records: List[Dict[str, Any]] = []
    for sdir in iter_dirs:
        train_fp = sdir / "train.parquet"
        test_fp = sdir / "test.parquet"
        if not train_fp.exists() or not test_fp.exists():
            continue

        df_train = _load_split(train_fp)
        df_test = _load_split(test_fp)

        y_train = df_train["sex"].map({"f": 0, "m": 1})
        y_test = df_test["sex"].map({"f": 0, "m": 1})

        drop_cols = ["sex"]
        if "individual_id" in df_train.columns:
            drop_cols.append("individual_id")
        X_train = df_train.drop(columns=drop_cols)
        X_test = df_test.drop(columns=drop_cols)
        X_train = X_train.select_dtypes(include=["number"]).copy()
        X_test = X_test.select_dtypes(include=["number"]).copy()

        if "Fold" in df_train.columns:
            fold_ids = df_train["Fold"].astype(int).to_numpy()
            X_train = X_train.drop(columns=["Fold"])
            cv = PredefinedSplit(fold_ids)
        else:
            cv = 5
        if "Fold" in df_test.columns:
            X_test = X_test.drop(columns=["Fold"])

        pipe = Pipeline([
            (
                "select",
                FeatureSelectionTransformer(method="forward", k=max_features),
            ),
            ("lda", LinearDiscriminantAnalysis()),
        ])

        if progress:
            folds = cv.get_n_splits() if hasattr(cv, "get_n_splits") else cv
            with tqdm_joblib(tqdm(desc=f"{sdir.name} CV", total=folds, leave=False)):
                cv_scores = cross_val_score(
                    pipe,
                    X_train,
                    y_train,
                    cv=cv,
                    scoring="balanced_accuracy",
                    n_jobs=n_jobs,
                )
        else:
            cv_scores = cross_val_score(
                pipe, X_train, y_train, cv=cv, scoring="balanced_accuracy",
                n_jobs=n_jobs
            )
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        bal = balanced_accuracy_score(y_test, y_pred)
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_test, y_pred, average="binary", zero_division=0
        )

        fem_i = mal_i = bal_i = maj_ct = maj_wr = maj_pct = None
        if "individual_id" in df_test.columns:
            fem_i, mal_i, bal_i = grouped_metrics.individual_accuracies(
                y_test.to_numpy(), y_pred, df_test["individual_id"]
            )
            maj_ct, maj_wr, maj_pct = grouped_metrics.individual_majority_stats(
                y_test.to_numpy(), y_pred, df_test["individual_id"]
            )

        dump(pipe, model_dir / f"{sdir.name}.joblib")

        if save_predictions:
            predict_simple_baseline(
                sdir.name, exp_dir, include_inference=True, reuse_csv=False
            )

        records.append(
            {
                "species": sdir.name,
                "cv_balanced_accuracy": float(cv_scores.mean()),
                "accuracy": acc,
                "balanced_accuracy": bal,
                "precision": prec,
                "recall": rec,
                "f1": f1,
                "female_individual_acc": fem_i,
                "male_individual_acc": mal_i,
                "balanced_individual_acc": bal_i,
                "maj_correct": maj_ct,
                "maj_wrong": maj_wr,
                "maj_pct": maj_pct,
            }
        )

    df = pd.DataFrame(records)
    df.to_csv(exp_dir / "raw_results.csv", index=False)
    return df



