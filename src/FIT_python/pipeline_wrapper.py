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

from FIT_python.config import (
    PROCESSED_SPLITS_DIR,
    NUMERIC_DIR,
    GLOBAL_RANDOM_SEED,
    DEBUG_MODE,
    PIPELINE_TARGETS,
    PIPELINE_STEPS,
    PIPELINE_PARAM_GRIDS,
    MODELS,
    METRICS,
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
        self.param_grids = PIPELINE_PARAM_GRIDS[target]
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
                k = PIPELINE_PARAM_GRIDS.get('feature_selection_k', None)
                step_list.append(('select', SelectKBest(k=k)))
            elif step == 'pca':
                n = PIPELINE_PARAM_GRIDS.get('pca_n_components', None)
                step_list.append(('pca', PCA(n_components=n, random_state=self.seed)))
            elif step == 'clf':
                step_list.append(('clf', self.models[model_name]))
            else:
                raise ValueError(f"Unknown pipeline step: {step}")
        return Pipeline(step_list)

    def run(self):
        """Run pipelines for all datasets and models for the given target."""
        for ds_folder in self.numeric_dir.iterdir():
            if not ds_folder.is_dir():
                continue
            dataset = ds_folder.name
            self.logger.info("Processing dataset: %s for target: %s", dataset, self.target)

            # Load numeric arrays
            X_train = np.load(ds_folder / f"X_train.npy", mmap_mode='r')
            y_train = np.load(ds_folder / f"y_{self.target}.npy", mmap_mode='r')
            X_test  = np.load(ds_folder / f"X_test.npy",  mmap_mode='r')
            y_test  = np.load(ds_folder / f"y_{self.target}.npy",  mmap_mode='r')

            # Load fold assignments from splits parquet
            df_train = pd.read_parquet(self.splits_dir / f"{dataset}_train.parquet")
            if 'fold' not in df_train.columns:
                raise KeyError("Column 'fold' not found in split file; cannot build PredefinedSplit")
            folds = df_train['fold'].to_numpy()
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
                    verbose=DEBUG_MODE
                )
                self.logger.info("Starting GridSearchCV for %s on %s", model_name, dataset)
                gs.fit(X_train, y_train)

                self.logger.info("Best params for %s: %s", model_name, gs.best_params_)
                self.logger.info("Best CV score for %s: %f", model_name, gs.best_score_)

                test_score = gs.score(X_test, y_test)
                self.logger.info("Test score for %s: %f", model_name, test_score)

    def build_and_run(self, target: str):
        self.run()

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: pipeline_wrapper.py <target>")
        sys.exit(1)
    target = sys.argv[1]
    wrapper = PipelineWrapper(target)
    wrapper.run()
