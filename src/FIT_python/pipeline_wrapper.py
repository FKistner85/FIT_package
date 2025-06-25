#!/usr/bin/env python3
"""Wrapper to build and execute sklearn pipelines for various targets."""

import sys
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.metrics import get_scorer
import joblib

from FIT_python.config import (
    PROCESSED_SPLITS_DIR,
    NUMERIC_DIR,
    GLOBAL_RANDOM_SEED,
    DEBUG_MODE,
    PIPELINE_TARGETS,
    PIPELINE_STEPS,
    BAYES_SPACES,
    MODELS,
    METRICS,
    RESULTS_DIR,
)

class PipelineWrapper:
    """Builds and runs sklearn Pipelines per target and per model."""

    def __init__(self, target: str):
        if target not in PIPELINE_TARGETS:
            raise ValueError(f"Unknown target: {target}")
        self.target = target
        self.seed = GLOBAL_RANDOM_SEED
        self.splits_dir = Path(PROCESSED_SPLITS_DIR)
        self.numeric_dir = Path(NUMERIC_DIR)
        self.steps = PIPELINE_STEPS  # e.g. ['scaler', 'select', 'pca', 'clf']
        # ``BAYES_SPACES`` contains search spaces per model; older code expected
        # ``PIPELINE_PARAM_GRIDS`` keyed by target.  Since only the ``sex``
        # target is supported, reuse the global BAYES_SPACES mapping.
        self.param_grids = BAYES_SPACES
        self.models = MODELS
        self.scoring = METRICS
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
        self.logger = logging.getLogger(__name__)

    def _build_pipeline(self, model_name: str) -> Pipeline:
        """Construct a sklearn Pipeline for a given model."""
        step_list = []
        for step in self.steps:
            if step == 'scaler':
                step_list.append(('scaler', StandardScaler()))
            elif step == 'select':
                # No configured feature count -> use all features
                k = None
                step_list.append(('select', SelectKBest(k=k)))
            elif step == 'pca':
                # Default to all components when not specified
                n = None
                step_list.append(('pca', PCA(n_components=n, random_state=self.seed)))
            elif step == 'clf':
                step_list.append(('clf', self.models[model_name]))
            else:
                raise ValueError(f"Unknown pipeline step: {step}")
        return Pipeline(step_list)

    def run_pipeline(self, filter_na: bool = False):
        """Run pipelines for all datasets and models for the given target."""
        results = []
        model_dir = RESULTS_DIR / "models"
        model_dir.mkdir(parents=True, exist_ok=True)

        for ds_folder in self.numeric_dir.iterdir():
            if not ds_folder.is_dir():
                continue
            dataset = ds_folder.name
            self.logger.info("Processing dataset: %s for target: %s", dataset, self.target)

            X_train = np.load(ds_folder / "X_train.npy", mmap_mode="r")
            y_train = np.load(ds_folder / f"y_{self.target}.npy", mmap_mode="r")
            X_test = np.load(ds_folder / "X_test.npy", mmap_mode="r")
            y_test = np.load(ds_folder / f"y_{self.target}.npy", mmap_mode="r")

            df_train = pd.read_parquet(self.splits_dir / f"{dataset}_train.parquet")
            df_test = pd.read_parquet(self.splits_dir / f"{dataset}_test.parquet")

            if filter_na:
                mask_tr = df_train["sex"].isin(["male", "female"])
                mask_te = df_test["sex"].isin(["male", "female"])
                df_train = df_train[mask_tr]
                df_test = df_test[mask_te]
                X_train = X_train[mask_tr]
                y_train = y_train[mask_tr]
                X_test = X_test[mask_te]
                y_test = y_test[mask_te]

            if "fold" not in df_train.columns:
                raise KeyError("Column 'fold' not found in split file; cannot build PredefinedSplit")
            folds = df_train["fold"].to_numpy()
            ps = PredefinedSplit(test_fold=folds)

            for model_name in self.models:
                self.logger.info("Building pipeline for model: %s", model_name)
                pipe = self._build_pipeline(model_name)
                param_grid = self.param_grids.get(model_name, {})
                self.logger.info("Param grid: %s", param_grid)

                gs = GridSearchCV(
                    estimator=pipe,
                    param_grid=param_grid,
                    cv=ps,
                    scoring=self.scoring,
                    n_jobs=-1,
                    verbose=DEBUG_MODE,
                )
                self.logger.info("Starting GridSearchCV for %s on %s", model_name, dataset)
                gs.fit(X_train, y_train)

                self.logger.info("Best params for %s: %s", model_name, gs.best_params_)
                self.logger.info("Best CV score for %s: %f", model_name, gs.best_score_)

                test_score = gs.score(X_test, y_test)
                self.logger.info("Test score for %s: %f", model_name, test_score)

                joblib.dump(gs.best_estimator_, model_dir / f"{dataset}_{model_name}.joblib")
                results.append({
                    "dataset": dataset,
                    "model": model_name,
                    "cv_score": gs.best_score_,
                    "test_score": test_score,
                    "best_params": gs.best_params_,
                })
        return results

    def run(self, filter_na: bool = False):
        return self.run_pipeline(filter_na=filter_na)

    def build_and_run(self, target: str):
        self.run()

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: pipeline_wrapper.py <target>")
        sys.exit(1)
    target = sys.argv[1]
    wrapper = PipelineWrapper(target)
    wrapper.run()
