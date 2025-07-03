# src/FIT_python/pipeline/pipeline_wrapper.py

from pathlib import Path
import pandas as pd
from joblib import Memory, dump
from time import perf_counter
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
)
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

# sklearn cache for pipelines
_cache_dir = Path(RESULTS_DATA_DIR) / "pipeline_cache"
memory = Memory(location=_cache_dir, verbose=0)

# in‐memory cache for loaded splits
_DATA_CACHE: dict[str, dict[str, pd.DataFrame]] = {}

def splits_available() -> bool:
    """Return True if at least one species folder in SPLITS_DIR has both train.parquet & test.parquet."""
    p = Path(SPLITS_DIR)
    if not p.exists():
        return False
    for d in p.iterdir():
        if d.is_dir() and (d/"train.parquet").exists() and (d/"test.parquet").exists():
            return True
    return False

def get_pipeline_steps(
    fs_method: str | None = None,
    fs_k: int | None = None,
    impute_method: str | None = None,
    outlier_method: str | None = None,
    scaler_method: str | None = None,
    reduce_pre_method: str | None = None,
    reduce_post_method: str | None = None
) -> list[tuple[str, object]]:
    steps: list[tuple[str, object]] = []
    # 1) Raw → numeric
    steps.append(("transform", NumericTransformer()))
    # 2) Imputation
    if impute_method == "miss_forest":
        steps.append(("impute", ImputationWrapper()))
    # 3) Outlier cleaning
    if outlier_method == "clip":
        steps.append(("outlier", OutlierCleanerTransformer(method="clip", lower_quantile=0.01, upper_quantile=0.99)))
    # 4) Scaling
    if scaler_method in ("standard", "robust"):
        steps.append(("scale", FeatureScalerTransformer(method=scaler_method)))
    # 5) Dimensionality reduction before selection
    if reduce_pre_method == "pca":
        steps.append(("reduce_pre", DimensionalityReducerTransformer(method="pca", n_components=10)))
    # 6) Feature selection
    if fs_method:
        steps.append(("select", FeatureSelectionTransformer(method=fs_method, k=fs_k)))
    # 7) Dimensionality reduction after selection
    if reduce_post_method == "pca":
        steps.append(("reduce_post", DimensionalityReducerTransformer(method="pca", n_components=10)))
    return steps

class PipelineWrapper:
    def __init__(
        self,
        model_keys: list[str] | None = None,
        fs_method: str | None = None,    # None = no selection
        fs_k: int | None = None,         # None = keep all
        impute_method: str | None = None,
        outlier_method: str | None = None,
        scaler_method: str | None = None,
        reduce_pre_method: str | None = None,
        reduce_post_method: str | None = None,
    ):
        # ensure output folder
        Path(RESULTS_DATA_DIR).mkdir(parents=True, exist_ok=True)

        self.model_keys      = model_keys or list(MODELS.keys())
        self.fs_method       = fs_method
        self.fs_k            = fs_k
        self.impute_method   = impute_method
        self.outlier_method  = outlier_method
        self.scaler_method   = scaler_method
        self.reduce_pre_method  = reduce_pre_method
        self.reduce_post_method = reduce_post_method

        self._model_dir = Path(RESULTS_DATA_DIR) / "models"
        self._model_dir.mkdir(parents=True, exist_ok=True)

    def run(self) -> pd.DataFrame:
        # 1) optionally regenerate splits
        if not splits_available():
            SplitWrapper().split_all()

        records: list[dict] = []

        # 2) for each species folder
        for sp in sorted(Path(SPLITS_DIR).iterdir()):
            if not sp.is_dir():
                continue
            train_f = sp / "train.parquet"
            test_f  = sp / "test.parquet"
            if not train_f.exists() or not test_f.exists():
                continue

            # 3) cache
            key = sp.name
            if key in _DATA_CACHE:
                df_tr, df_te = _DATA_CACHE[key]["train"], _DATA_CACHE[key]["test"]
            else:
                df_tr = (pd.read_parquet(train_f)
                            .dropna(subset=["sex"])
                            .query("sex in ['f','m']"))
                df_te = (pd.read_parquet(test_f)
                            .dropna(subset=["sex"])
                            .query("sex in ['f','m']"))
                _DATA_CACHE[key] = {"train": df_tr, "test": df_te}

            # 4) labels
            y_tr = df_tr["sex"].map({"f":0, "m":1})
            y_te = df_te["sex"].map({"f":0, "m":1})

            # 5) fixed CV on pre-computed Fold
            cv = PredefinedSplit(test_fold=df_tr["Fold"].values)
            X_tr = df_tr.drop(columns=["Fold"])
            X_te = df_te

            # 6) per model
            for mk in self.model_keys:
                model = MODELS[mk]
                steps = get_pipeline_steps(
                    fs_method=self.fs_method,
                    fs_k=self.fs_k,
                    impute_method=self.impute_method,
                    outlier_method=self.outlier_method,
                    scaler_method=self.scaler_method,
                    reduce_pre_method=self.reduce_pre_method,
                    reduce_post_method=self.reduce_post_method,
                )
                steps.append(("clf", model))
                pipe = Pipeline(steps, memory=memory)

                # a) CV
                try:
                    accs = cross_val_score(pipe, X_tr, y_tr, cv=cv, scoring="accuracy", n_jobs=1)
                    bals = cross_val_score(pipe, X_tr, y_tr, cv=cv, scoring="balanced_accuracy", n_jobs=1)
                    cv_acc, cv_bal = float(accs.mean()), float(bals.mean())
                except Exception:
                    cv_acc = cv_bal = None

                # b) fit + timing
                t0_all = perf_counter()
                times = {}
                X = X_tr.copy()
                for name, step in pipe.steps[:-1]:
                    t0 = perf_counter()
                    X = step.fit_transform(X, y_tr) if hasattr(step,"fit_transform") else step.fit(X,y_tr).transform(X)
                    times[f"time_{name}"] = perf_counter() - t0

                clf = pipe.steps[-1][1]
                t0 = perf_counter()
                clf.fit(X, y_tr)
                times["time_classifier"] = perf_counter() - t0

                t0 = perf_counter()
                y_pred = pipe.predict(X_te)
                times["time_predict"] = perf_counter() - t0
                total_time = perf_counter() - t0_all

                # c) test metrics
                test_acc = accuracy_score(y_te, y_pred)
                test_bal = balanced_accuracy_score(y_te, y_pred)
                report   = classification_report(y_te, y_pred, output_dict=True)

                sel = pipe.named_steps.get("select", None)
                feats    = getattr(sel, "selected_features_", None)
                ranking  = getattr(sel, "feature_ranking_",  None)

                rec = {
                    "species":               key,
                    "model":                 mk,
                    "cv_accuracy":           cv_acc,
                    "cv_balanced_accuracy":  cv_bal,
                    "test_accuracy":         float(test_acc),
                    "test_balanced_accuracy":float(test_bal),
                    "time_total":            total_time,
                    "classification_report": report,
                    "fs_method":             self.fs_method,
                    "fs_k":                  self.fs_k,
                    "selected_features":     feats,
                    "feature_ranking":       ranking,
                    "impute_method":         self.impute_method,
                    "outlier_method":        self.outlier_method,
                    "scaler_method":         self.scaler_method,
                    "reduce_pre_method":     self.reduce_pre_method,
                    "reduce_post_method":    self.reduce_post_method,
                }
                rec.update(times)
                records.append(rec)

        df_new = pd.DataFrame(records)

        raw_out  = Path(RESULTS_DATA_DIR) / "raw_results.csv"
        best_out = Path(RESULTS_DATA_DIR) / "best_results.csv"

        # append raw
        if raw_out.exists():
            df_raw = pd.concat([pd.read_csv(raw_out), df_new], ignore_index=True)
        else:
            df_raw = df_new.copy()
        df_raw.to_csv(raw_out, index=False)

        # update best per species
        if best_out.exists():
            df_all = pd.concat([pd.read_csv(best_out), df_new], ignore_index=True)
        else:
            df_all = df_new.copy()
        df_best = (
            df_all
            .sort_values("test_balanced_accuracy", ascending=False)
            .drop_duplicates(subset=["species"], keep="first")
            .reset_index(drop=True)
        )
        df_best.to_csv(best_out, index=False)

        # dump final best pipelines on full training set
        for _, row in df_best.iterrows():
            df_t = _DATA_CACHE[row["species"]]["train"]
            y_t  = df_t["sex"].map({"f":0,"m":1})
            X_t  = df_t.drop(columns=["Fold","sex"])

            steps = get_pipeline_steps(
                fs_method        = row["fs_method"],
                fs_k             = row["fs_k"],
                impute_method    = row["impute_method"],
                outlier_method   = row["outlier_method"],
                scaler_method    = row["scaler_method"],
                reduce_pre_method= row["reduce_pre_method"],
                reduce_post_method=row["reduce_post_method"],
            )
            steps.append(("clf", MODELS[row["model"]]))
            final_pipe = Pipeline(steps)
            final_pipe.fit(X_t, y_t)
            dump(final_pipe, self._model_dir/f"{row['species']}.joblib")

        return df_raw
