#!/usr/bin/env python3
"""Hyperparameter utilities combining Optuna tuning and model constructors."""

import numpy as np
import pandas as pd
import optuna
from typing import Union, Callable, Tuple
from sklearn.model_selection import PredefinedSplit
from sklearn.metrics import get_scorer

from FIT_python.config import GLOBAL_RANDOM_SEED

def get_model_constructor(name: str, seed: int) -> Callable[[optuna.trial.Trial], object]:
    """Return a constructor for the requested model."""
    from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
    from sklearn.svm import SVC
    from sklearn.linear_model import LogisticRegression
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from xgboost import XGBClassifier
    from lightgbm import LGBMClassifier

    if name == "Random Forest":
        def constructor(trial):
            return RandomForestClassifier(
                n_estimators=trial.suggest_int("n_estimators", 50, 300),
                max_depth=trial.suggest_int("max_depth", 2, 20),
                min_samples_split=trial.suggest_int("min_samples_split", 2, 10),
                random_state=seed
            )
    elif name == "Extra Trees":
        def constructor(trial):
            return ExtraTreesClassifier(
                n_estimators=trial.suggest_int("n_estimators", 50, 300),
                max_depth=trial.suggest_int("max_depth", 2, 20),
                random_state=seed
            )
    elif name == "SVM (RBF Kernel)":
        def constructor(trial):
            return SVC(
                C=trial.suggest_float("C", 1e-3, 10.0, log=True),
                gamma=trial.suggest_float("gamma", 1e-4, 1.0, log=True),
                probability=True,
                random_state=seed
            )
    elif name == "Logistic Regression":
        def constructor(trial):
            return LogisticRegression(
                C=trial.suggest_float("C", 1e-3, 10.0, log=True),
                solver="lbfgs",
                max_iter=1000,
                random_state=seed
            )
    elif name == "XGBoost":
        def constructor(trial):
            return XGBClassifier(
                n_estimators=trial.suggest_int("n_estimators", 50, 300),
                learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                max_depth=trial.suggest_int("max_depth", 3, 10),
                eval_metric="mlogloss",
                random_state=seed
            )
    elif name == "LightGBM":
        def constructor(trial):
            return LGBMClassifier(
                n_estimators=trial.suggest_int("n_estimators", 50, 300),
                learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                max_depth=trial.suggest_int("max_depth", 3, 10),
                random_state=seed
            )
    elif name == "LDA":
        def constructor(_: optuna.trial.Trial):
            return LinearDiscriminantAnalysis()
    else:
        raise ValueError(f"Unknown model name: {name}")

    return constructor

def optimize_hyperparameters(
    model_name: str,
    X: pd.DataFrame,
    y: Union[pd.Series, np.ndarray],
    individual_ids: pd.Series,
    fold_assignments: pd.DataFrame,
    scoring: Union[str, Callable],
    n_trials: int = 30,
    seed: int = None
) -> Tuple[dict, float, pd.DataFrame]:
    """Run Optuna hyperparameter search on predefined individual folds."""
    if not isinstance(y, pd.Series):
        y = pd.Series(y, index=X.index)

    if seed is None:
        seed = GLOBAL_RANDOM_SEED

    if not {'individual','Fold'}.issubset(fold_assignments.columns):
        raise ValueError("fold_assignments must contain 'individual' and 'Fold'")
    fold_map = dict(zip(fold_assignments['individual'], fold_assignments['Fold']))
    test_folds = individual_ids.map(fold_map)
    if test_folds.isnull().any():
        missing = individual_ids[test_folds.isnull()].unique()
        raise ValueError(f"Missing fold assignments for individuals: {missing}")
    ps = PredefinedSplit(test_folds.astype(int).values)

    scorer_fn = get_scorer(scoring) if isinstance(scoring, str) else scoring
    constructor = get_model_constructor(model_name, seed)

    def objective(trial: optuna.trial.Trial) -> float:
        model = constructor(trial)
        scores = []
        for train_idx, val_idx in ps.split():
            X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
            model.fit(X_tr, y_tr)
            scores.append(scorer_fn(model, X_val, y_val))
        return float(np.nanmean(scores))

    direction = 'minimize' if (isinstance(scoring, str) and scoring.startswith('neg_')) else 'maximize'
    study = optuna.create_study(direction=direction, sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials)

    return study.best_params, study.best_value, study.trials_dataframe()

def get_best_hyperparameters(
    model_name: str,
    X: pd.DataFrame,
    y: Union[pd.Series, np.ndarray],
    individual_ids: pd.Series,
    fold_assignments: pd.DataFrame,
    scoring: Union[str, Callable],
    seed: int = None
) -> dict:
    """Return only best_params from optimize_hyperparameters."""
    best_params, _, _ = optimize_hyperparameters(
        model_name=model_name,
        X=X,
        y=y,
        individual_ids=individual_ids,
        fold_assignments=fold_assignments,
        scoring=scoring,
        n_trials=30,
        seed=seed
    )
    return best_params
