#!/usr/bin/env python3
"""Model comparison for target 'sex' across datasets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier

try:
    from catboost import CatBoostClassifier
except Exception:  # pragma: no cover - optional dependency
    CatBoostClassifier = None  # type: ignore
try:
    from lightgbm import LGBMClassifier
except Exception:  # pragma: no cover
    LGBMClassifier = None  # type: ignore
try:
    from xgboost import XGBClassifier
except Exception:  # pragma: no cover
    XGBClassifier = None  # type: ignore

from FIT_python.config import (
    NUMERIC_DIR,
    SPLITS_DIR,
    RESULTS_DIR,
    DEBUG_MODE,
    VALIDATION_MODE,
    NUM_KFOLDS,
    normalize_dataset_name,
)


def load_numpy(path: Path) -> np.ndarray:
    if not path.exists():
        if DEBUG_MODE:
            raise FileNotFoundError(f"Missing required file: {path}")
        else:
            print(f"Warning: skipping missing {path}")
            raise RuntimeError("skip")
    return np.load(path, allow_pickle=True)


def load_optional(path: Path) -> np.ndarray | None:
    if not path.exists():
        return None
    return np.load(path, allow_pickle=True)


def last_two_to_labels(y: np.ndarray) -> np.ndarray:
    """Convert one-hot encoded array to binary labels using last two columns."""
    if y.shape[1] < 2:
        raise ValueError("y array must have at least two columns for sex")
    return np.argmax(y[:, -2:], axis=1)


def build_models() -> Dict[str, Dict[str, Any]]:
    models: Dict[str, Dict[str, Any]] = {
        "LogisticRegression": {
            "estimator": LogisticRegression(max_iter=1000, solver="liblinear"),
            "params": {"C": [0.1, 1, 10], "penalty": ["l1", "l2"]},
        },
        "SVM": {
            "estimator": SVC(probability=True),
            "params": {"C": [0.1, 1, 10], "kernel": ["linear", "rbf"]},
        },
        "RandomForest": {
            "estimator": RandomForestClassifier(random_state=0),
            "params": {"n_estimators": [50, 100], "max_depth": [None, 10]},
        },
        "kNN": {
            "estimator": KNeighborsClassifier(),
            "params": {"n_neighbors": [3, 5, 7]},
        },
        "Lasso": {
            "estimator": SGDClassifier(loss="log_loss", penalty="l1"),
            "params": {"alpha": [0.0001, 0.001, 0.01]},
        },
        "LDA": {"estimator": LinearDiscriminantAnalysis(), "params": {}},
    }
    if CatBoostClassifier:
        models["CatBoost"] = {
            "estimator": CatBoostClassifier(verbose=0),
            "params": {"iterations": [50, 100], "depth": [3, 6]},
        }
    if LGBMClassifier:
        models["LightGBM"] = {
            "estimator": LGBMClassifier(),
            "params": {"num_leaves": [31, 64], "learning_rate": [0.1, 0.01]},
        }
    if XGBClassifier:
        models["XGBoost"] = {
            "estimator": XGBClassifier(eval_metric="logloss"),
            "params": {"n_estimators": [50, 100], "max_depth": [3, 6]},
        }
    # Optional simple models
    models["AdaBoost"] = {
        "estimator": AdaBoostClassifier(random_state=0),
        "params": {"n_estimators": [50, 100]},
    }
    models["GaussianNB"] = {"estimator": GaussianNB(), "params": {}}
    return models


def evaluate_model(model, X, y) -> Dict[str, float]:
    pred = model.predict(X)
    metrics = {
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
    }
    try:
        prob = model.predict_proba(X)[:, 1]
        metrics["auc"] = roc_auc_score(y, prob)
    except Exception:
        metrics["auc"] = float("nan")
    return metrics


def main() -> None:
    results: List[Dict[str, Any]] = []
    for ds_folder in NUMERIC_DIR.iterdir():
        if not ds_folder.is_dir():
            continue
        dataset = normalize_dataset_name(ds_folder.name)
        print(f"\n=== Dataset: {dataset} ===")
        try:
            X_train = load_numpy(ds_folder / "X_train.npy")
            y_train = load_numpy(ds_folder / "y_train.npy")
        except RuntimeError:
            continue
        y_train = last_two_to_labels(y_train)
        X_test = load_optional(ds_folder / "X_test.npy")
        y_test = load_optional(ds_folder / "y_test.npy")
        if y_test is not None:
            y_test = last_two_to_labels(y_test)

        # Validation setup
        if VALIDATION_MODE == "external_folds":
            folds_path = SPLITS_DIR / dataset / "folds.parquet"
            if not folds_path.exists():
                msg = f"Missing folds file: {folds_path}"
                if DEBUG_MODE:
                    raise FileNotFoundError(msg)
                else:
                    print("Skipping:", msg)
                    continue
            folds_df = pd.read_parquet(folds_path)
            if "Fold" not in folds_df.columns:
                msg = f"No 'Fold' column in {folds_path}"
                if DEBUG_MODE:
                    raise ValueError(msg)
                else:
                    print("Skipping:", msg)
                    continue
            test_fold = folds_df["Fold"].to_numpy()
            cv = PredefinedSplit(test_fold)
        else:
            cv = 3

        models = build_models()
        scoring = {
            "accuracy": "accuracy",
            "precision": "precision",
            "recall": "recall",
            "f1": "f1",
            "roc_auc": "roc_auc",
        }
        for name, cfg in models.items():
            print(f"  Model: {name}")
            grid = GridSearchCV(
                cfg["estimator"],
                cfg["params"],
                cv=cv,
                scoring=scoring,
                refit="accuracy",
                n_jobs=-1,
            )
            try:
                grid.fit(X_train, y_train)
            except Exception as exc:
                print(f"    Failed: {exc}")
                continue
            best = grid.best_estimator_
            cv_metrics = {
                metric: grid.cv_results_[f"mean_test_{metric}"][grid.best_index_]
                for metric in ["accuracy", "precision", "recall", "f1", "roc_auc"]
            }
            row = {
                "Dataset": dataset,
                "Model": name,
                "BestParams": json.dumps(grid.best_params_),
                "CV_Acc": cv_metrics["accuracy"],
                "CV_Prec": cv_metrics["precision"],
                "CV_Rec": cv_metrics["recall"],
                "CV_F1": cv_metrics["f1"],
                "CV_AUC": cv_metrics["roc_auc"],
                "Test_Acc": None,
                "Test_Prec": None,
                "Test_Rec": None,
                "Test_F1": None,
                "Test_AUC": None,
            }
            if X_test is not None and y_test is not None:
                test_metrics = evaluate_model(best, X_test, y_test)
                row.update({
                    "Test_Acc": test_metrics["accuracy"],
                    "Test_Prec": test_metrics["precision"],
                    "Test_Rec": test_metrics["recall"],
                    "Test_F1": test_metrics["f1"],
                    "Test_AUC": test_metrics.get("auc"),
                })
            results.append(row)

    if not results:
        print("No results to save.")
        return
    df = pd.DataFrame(results)
    out_path = Path("data/results/model_comparison_sex.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Saved results to {out_path}")


if __name__ == "__main__":
    main()
