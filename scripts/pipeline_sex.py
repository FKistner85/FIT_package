#!/usr/bin/env python3
"""Bayesian optimised sklearn pipeline for the 'sex' target."""

from pathlib import Path
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.model_selection import PredefinedSplit
from skopt import BayesSearchCV
import joblib

from FIT_python.old_files.scaler_wrapper import ScalerWrapper
from FIT_python.old_files.feature_wrapper import FeatureSelector
from FIT_python.old_files.dim_reduction_wrapper import DimReducer
from FIT_python.config import (
    PROCESSED_SPLITS_DIR,
    DEFAULT_TARGETS,
    PIPELINE_TARGETS,
    SCALER_METHODS,
    FEATURE_SELECTION_METHODS,
    DIM_REDUCTION_METHODS,
    DIM_PARAMS,
    MODELS,
    BAYES_SPACES,
    HYPER_N_TRIALS,
    METRICS,
    RESULTS_DATA_DIR,
    GLOBAL_RANDOM_SEED,
)


def run_for_dataset(ds: str) -> None:
    df_train = pd.read_parquet(PROCESSED_SPLITS_DIR / f"{ds}_train.parquet")
    df_train = df_train[df_train["sex"].isin(["male", "female"])]
    X_train = df_train.drop(columns=DEFAULT_TARGETS).to_numpy()
    y_train = df_train["sex"].to_numpy()
    folds = df_train["fold"].to_numpy()
    ps = PredefinedSplit(test_fold=folds)

    pipe = Pipeline([
        ("scaler", ScalerWrapper(scaler_type=SCALER_METHODS[0])),
        ("feature_selection", FeatureSelector(methods=FEATURE_SELECTION_METHODS)),
        ("dim_reduction", DimReducer(methods=DIM_REDUCTION_METHODS, params=DIM_PARAMS)),
        ("clf", None),
    ])

    for name, estimator in MODELS.items():
        pipe.set_params(clf=estimator)
        opt = BayesSearchCV(
            estimator=pipe,
            search_spaces=BAYES_SPACES.get(name, {}),
            cv=ps,
            n_iter=HYPER_N_TRIALS,
            scoring=METRICS["sex"],
            n_jobs=-1,
            random_state=GLOBAL_RANDOM_SEED,
        )
        opt.fit(X_train, y_train)

        df_test = pd.read_parquet(PROCESSED_SPLITS_DIR / f"{ds}_test.parquet")
        df_test = df_test[df_test["sex"].isin(["male", "female"])]
        X_test = df_test.drop(columns=DEFAULT_TARGETS).to_numpy()
        y_test = df_test["sex"].to_numpy()
        test_score = opt.score(X_test, y_test)

        print(f"[REPORT] {ds} | {name} | CV={opt.best_score_:.3f} | Test={test_score:.3f}")
        out = RESULTS_DATA_DIR / f"pipeline_sex_{ds}_{name}.joblib"
        out.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(opt.best_estimator_, out)


def main() -> int:
    datasets = {p.stem.rsplit("_", 1)[0] for p in PROCESSED_SPLITS_DIR.glob("*_train.parquet")}
    for ds in sorted(datasets):
        run_for_dataset(ds)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
