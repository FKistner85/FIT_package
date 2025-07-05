# src/FIT_python/pipeline/pipeline_wrapper.py

"""Run sklearn pipelines on all species using balanced accuracy."""

from pathlib import Path
import pandas as pd
from joblib import Memory
from time import perf_counter
from sklearn.pipeline import Pipeline
from sklearn.metrics import balanced_accuracy_score, classification_report
from sklearn.model_selection import PredefinedSplit, cross_val_score

from FIT_python.config import SPLITS_DIR, RESULTS_DATA_DIR
from FIT_python.step_02_a_splitting_train_test.wrapper import SplitWrapper
from .transform_wrapper import NumericTransformer
from .imputation_wrapper import ImputationWrapper
from .outlier_wrapper import OutlierCleanerTransformer
from .feature_scaler_wrapper import FeatureScalerTransformer
from .feature_selection_wrapper import FeatureSelectionTransformer
from .dimensionality_reduction_wrapper import DimensionalityReducerTransformer
from .models import MODELS

_cache_dir = Path(RESULTS_DATA_DIR) / "pipeline_cache"
memory = Memory(location=_cache_dir, verbose=0)

def get_pipeline_steps(
    fs_method: str = "forward",
    fs_k: int = 20,
    impute_method: str | None = "miss_forest",
    outlier_method: str | None = "clip",
    scaler_method: str | None = "standard",
    reduce_pre_method: str | None = None,
    reduce_post_method: str | None = None
) -> list[tuple[str, object]]:
    steps: list[tuple[str, object]] = []
    steps.append(("transform", NumericTransformer()))
    if impute_method == "miss_forest":
        steps.append(("impute", ImputationWrapper()))
    if outlier_method == "clip":
        steps.append((
            "outlier",
            OutlierCleanerTransformer(method="clip", lower_quantile=0.01, upper_quantile=0.99)
        ))
    if scaler_method in ("standard", "robust"):
        steps.append((
            "scale",
            FeatureScalerTransformer(method=scaler_method)
        ))
    if reduce_pre_method == "pca":
        steps.append((
            "reduce_pre",
            DimensionalityReducerTransformer(method="pca", n_components=10)
        ))
    steps.append((
        "select",
        FeatureSelectionTransformer(method=fs_method, k=fs_k)
    ))
    if reduce_post_method == "pca":
        steps.append((
            "reduce_post",
            DimensionalityReducerTransformer(method="pca", n_components=10)
        ))
    return steps

class PipelineWrapper:
    def __init__(
        self,
        model_keys: list[str] | None = None,
        fs_method: str = 'forward',
        fs_k: int = 20,
        impute_method: str | None = 'miss_forest',
        outlier_method: str | None = 'clip',
        scaler_method: str | None = 'standard',
        reduce_pre_method: str | None = None,
        reduce_post_method: str | None = None
    ):
        RESULTS_DATA_DIR.mkdir(parents=True, exist_ok=True)
        _cache_dir.mkdir(parents=True, exist_ok=True)

        self.model_keys = model_keys or list(MODELS.keys())
        self.fs_method = fs_method
        self.fs_k = fs_k
        self.impute_method = impute_method
        self.outlier_method = outlier_method
        self.scaler_method = scaler_method
        self.reduce_pre_method = reduce_pre_method
        self.reduce_post_method = reduce_post_method

    def run(self) -> pd.DataFrame:
        SplitWrapper().split_all()
        records: list[dict] = []

        for species_dir in sorted(Path(SPLITS_DIR).iterdir()):
            if not species_dir.is_dir(): continue

            train_p = species_dir / 'train.parquet'
            test_p  = species_dir / 'test.parquet'
            if not train_p.exists() or not test_p.exists(): continue

            df_train = pd.read_parquet(train_p).dropna(subset=['sex'])
            df_test  = pd.read_parquet(test_p).dropna(subset=['sex'])
            df_train = df_train[df_train['sex'].isin(['f','m'])]
            df_test  = df_test[df_test['sex'].isin(['f','m'])]

            y_train = df_train['sex'].map({'f':0,'m':1})
            y_test  = df_test['sex'].map({'f':0,'m':1})

            ps = PredefinedSplit(test_fold=df_train['Fold'].values)
            X_train = df_train.drop(columns=['Fold'])
            X_test  = df_test

            for key in self.model_keys:
                model = MODELS[key]
                steps = get_pipeline_steps(
                    fs_method=self.fs_method,
                    fs_k=self.fs_k,
                    impute_method=self.impute_method,
                    outlier_method=self.outlier_method,
                    scaler_method=self.scaler_method,
                    reduce_pre_method=self.reduce_pre_method,
                    reduce_post_method=self.reduce_post_method
                )
                steps.append(('classifier', model))
                pipe = Pipeline(steps, memory=memory)

                # --- CV ---
                try:
                    cv_scores = cross_val_score(
                        pipe,
                        X_train,
                        y_train,
                        cv=ps,
                        scoring='balanced_accuracy',
                        n_jobs=-1,
                    )
                    cv_mean = float(cv_scores.mean())
                except Exception:
                    cv_mean = None

                # --- Final Fit + Timing pro Schritt ---
                times: dict[str, float] = {}
                X = X_train.copy()
                for name, step in pipe.steps[:-1]:
                    t0 = perf_counter()
                    # fit_transform für alle außer classifier
                    if hasattr(step, "fit_transform"):
                        X = step.fit_transform(X, y_train)
                    else:
                        X = step.fit(X, y_train).transform(X)
                    times[f"time_{name}"] = perf_counter() - t0

                # classifier
                clf = pipe.steps[-1][1]
                t0 = perf_counter()
                clf.fit(X, y_train)
                times["time_classifier"] = perf_counter() - t0

                # Predict
                t0 = perf_counter()
                # Vorhersage durch gesamte transform-Kette + classifier
                y_pred = pipe.predict(X_test)
                times["time_predict"] = perf_counter() - t0

                test_acc = float(balanced_accuracy_score(y_test, y_pred))
                report   = classification_report(y_test, y_pred, output_dict=True)

                # Record zusammenbauen
                record = {
                    'species': species_dir.name,
                    'model': key,
                    'cv_balanced_accuracy': cv_mean,
                    'test_balanced_accuracy': test_acc,
                    'classification_report': report
                }
                # Pipeline-Parametrisierung
                record.update({
                    'fs_method': self.fs_method,
                    'fs_k': self.fs_k,
                    'impute_method': self.impute_method,
                    'outlier_method': self.outlier_method,
                    'scaler_method': self.scaler_method,
                    'reduce_pre_method': self.reduce_pre_method,
                    'reduce_post_method': self.reduce_post_method
                })
                # Zeiten hinzufügen
                record.update(times)

                records.append(record)

        df_results = pd.DataFrame(records)
        out = Path(RESULTS_DATA_DIR) / 'model_results.csv'
        df_results.to_csv(out, index=False)
        return df_results
