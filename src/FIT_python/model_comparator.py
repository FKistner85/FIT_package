"""Simple model comparison on scaled datasets."""

from pathlib import Path
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV, PredefinedSplit, StratifiedKFold
from sklearn.pipeline import Pipeline

import FIT_python.config as config
from FIT_python.config import DEFAULT_TARGETS, OTTER_META_COLS
from FIT_python.path_utils import scaled_path
import sys


class FeatureSubsetter(BaseEstimator, TransformerMixin):
    """Simple transformer that selects columns by integer indices."""

    def __init__(self, indices: List[int]):
        self.indices = indices

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[:, self.indices]


class DimReducerTransform(BaseEstimator, TransformerMixin):
    """Dimensionality reduction wrapper supporting PCA only."""

    def __init__(self, n_components: int = 2):
        self.n_components = n_components
        self.model: Optional[PCA] = None

    def fit(self, X, y=None):
        self.model = PCA(n_components=self.n_components, random_state=0)
        self.model.fit(X)
        return self

    def transform(self, X):
        return self.model.transform(X)


class ModelWrapper:
    """Train a single pipeline with GridSearchCV."""

    def __init__(self, model, param_grid: Dict[str, Any], scoring: str) -> None:
        self.model = model
        self.param_grid = param_grid
        self.scoring = scoring
        self.best_pipeline_ = None
        self.best_score_ = None
        self.best_params_ = None

    def fit(self, X_train, y_train, cv, fs_idx: List[int], n_components: int) -> None:
        steps = [
            ("select", FeatureSubsetter(fs_idx)),
            ("dim", DimReducerTransform(n_components)),
            ("clf", self.model),
        ]
        pipe = Pipeline(steps)
        grid = GridSearchCV(
            pipe,
            self.param_grid,
            cv=cv,
            scoring=self.scoring,
            n_jobs=-1,
        )
        grid.fit(X_train, y_train)
        self.best_pipeline_ = grid.best_estimator_
        self.best_score_ = grid.best_score_
        self.best_params_ = grid.best_params_

    def score(self, X_test, y_test) -> float:
        return self.best_pipeline_.score(X_test, y_test)


class LegacyModelComparator:
    """Train small models on each scaled dataset and save accuracies."""

    def compare_all(self) -> int:
        if not config.SCALED_DIR.exists():
            msg = f"Required file not found: {config.SCALED_DIR}"
            print(f"[ERROR] {msg}", file=sys.stderr)
            if config.DEBUG_MODE:
                raise FileNotFoundError(msg)
            return 1
        config.RESULTS_DATA_DIR.mkdir(parents=True, exist_ok=True)
        datasets = {
            p.stem.rsplit("_", 1)[0]
            for p in config.SCALED_DIR.glob("*_train.parquet")
        }
        rows = []
        for dataset in datasets:
            train_p = scaled_path(dataset, "train")
            test_p = scaled_path(dataset, "test")
            if not train_p.exists() or not test_p.exists():
                missing = train_p if not train_p.exists() else test_p
                msg = f"Required file not found: {missing}"
                print(f"[ERROR] {msg}", file=sys.stderr)
                if config.DEBUG_MODE:
                    raise FileNotFoundError(msg)
                continue

            df_train = pd.read_parquet(train_p)
            df_test = pd.read_parquet(test_p)
            meta_cols = OTTER_META_COLS if "otter" in dataset.lower() else ["id"]
            feature_cols = [
                c for c in df_train.columns if c not in meta_cols + DEFAULT_TARGETS
            ]
            X_train, y_train = df_train[feature_cols], df_train["sex"]
            X_test, y_test = df_test[feature_cols], df_test["sex"]
            models = {
                "LogReg": LogisticRegression(max_iter=1000),
                "SVM": SVC(),
            }
            for name, model in models.items():
                model.fit(X_train, y_train)
                pred = model.predict(X_test)
                acc = accuracy_score(y_test, pred)
                rows.append({"Dataset": dataset, "Model": name, "Accuracy": acc})
                print(f"[REPORT] {dataset} {name} accuracy={acc:.3f}")

        out_path = config.RESULTS_DATA_DIR / "model_comparison.csv"
        pd.DataFrame(rows).to_csv(out_path, index=False)
        print(f"[SUCCESS] results written to {out_path}")
        return 0


class ModelComparator:
    """Grid-search based comparison over scalers and feature methods."""

    def run(self) -> int:
        results = []
        for scaler_name in config.SCALER_PARAMS.keys():
            num_root = config.NUMERIC_DIR / scaler_name
            if not num_root.exists():
                continue
            for ds_folder in num_root.iterdir():
                if not ds_folder.is_dir():
                    continue
                dataset = ds_folder.name
                X_train = np.load(ds_folder / "X_train.npy")
                y_train = np.load(ds_folder / "y_train.npy").ravel()
                X_test = np.load(ds_folder / "X_test.npy")
                y_test = np.load(ds_folder / "y_test.npy").ravel()

                df_train = pd.read_parquet(
                    config.SCALED_DIR / scaler_name / f"{dataset}_train.parquet"
                )
                meta_cols = OTTER_META_COLS if "otter" in dataset.lower() else ["id"]
                feature_cols = [c for c in df_train.columns if c not in meta_cols + DEFAULT_TARGETS]
                df_train_feat = pd.DataFrame(X_train, columns=feature_cols)

                fs_results = run_feature_selection_methods(
                    df_train_feat,
                    pd.Series(y_train),
                    methods=config.FS_DEFAULT_METHODS,
                    target_feature_counts=config.FS_TARGET_FEATURE_COUNTS,
                    verbose=False,
                )
                for fs_name, feats in fs_results.items():
                    fs_idx = [feature_cols.index(f) for f in feats]
                    for model_name, model in config.MODELS.items():
                        param_grid = config.PIPELINE_PARAM_GRIDS["sex"].get(model_name, {})
                        wrapper = ModelWrapper(model, param_grid, config.METRICS["sex"])
                        cv = StratifiedKFold(n_splits=3)
                        wrapper.fit(X_train, y_train, cv, fs_idx, config.N_COMPONENTS)
                        score = wrapper.score(X_test, y_test)
                        print(
                            f"[REPORT] {dataset} {scaler_name} {fs_name} {model_name} test={score:.3f}"
                        )
                        results.append(
                            {
                                "dataset": dataset,
                                "scaler": scaler_name,
                                "fs": fs_name,
                                "model": model_name,
                                "score": score,
                                "best_params": wrapper.best_params_,
                                "cv_score": wrapper.best_score_,
                            }
                        )

        if results:
            out = pd.DataFrame(results)
            out_path = config.RESULTS_DATA_DIR / "pipeline_sex_results.csv"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out.to_csv(out_path, index=False)
            print(f"[SUCCESS] results written to {out_path}")
        return 0
