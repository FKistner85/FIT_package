# src/FIT_python/pipeline/pipeline_wrapper.py

from pathlib import Path
import pandas as pd
import numpy as np
from typing import Optional, Union, List
from joblib import Memory, dump
from time import perf_counter

from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
)
from sklearn.model_selection import (
    PredefinedSplit,
    cross_val_score,
    KFold,
)

from FIT_python.config import (
    SPLITS_DIR,
    RESULTS_DATA_DIR,
    GLOBAL_RANDOM_SEED,
    NUM_FOLDS,
    GROUP_COL,
)
# ← Hier die Utility-Funktionen importieren:
from FIT_python.pipeline.split_utils import (
    splits_available,
    _make_folds,
    ensure_valid_splits,
)
from FIT_python.pipeline.split_wrapper import SplitWrapper

from .transform_wrapper import NumericTransformer
from .imputation_wrapper import ImputationWrapper
from .outlier_wrapper import OutlierCleanerTransformer
from .feature_scaler_wrapper import FeatureScalerTransformer
from .feature_selection_wrapper import FeatureSelectionTransformer
from .dimensionality_reduction_wrapper import DimensionalityReducerTransformer
from .models import MODELS

# Cache für sklearn-Pipelines
_cache_dir = Path(RESULTS_DATA_DIR) / "pipeline_cache"
memory = Memory(location=_cache_dir, verbose=0)

# In-Memory Cache der geladenen Splits
_DATA_CACHE: dict[str, dict[str, pd.DataFrame]] = {}

# Erlaubte FS-Methoden
_ALLOWED_FS = ["forward", "random_forest", "variance", "univariate", "lasso"]


def get_pipeline_steps(
    fs_method: Optional[str] = None,
    fs_k: Optional[int] = None,
    impute_method: Optional[str] = None,
    outlier_method: Optional[str] = None,
    scaler_method: Optional[str] = None,
    reduce_pre_method: Optional[str] = None,
    reduce_post_method: Optional[str] = None
) -> list[tuple[str, object]]:
    steps: list[tuple[str, object]] = []
    # 1) Numeric
    steps.append(("transform", NumericTransformer()))
    # 2) Impute
    if impute_method == "miss_forest":
        steps.append(("impute", ImputationWrapper()))
    # 3) Outlier
    if outlier_method == "clip":
        steps.append(("outlier", OutlierCleanerTransformer(
            method="clip", lower_quantile=0.01, upper_quantile=0.99)))
    # 4) Scale
    if scaler_method in ("standard", "robust"):
        steps.append(("scale", FeatureScalerTransformer(method=scaler_method)))
    # 5) Pre-DimRed
    if reduce_pre_method == "pca":
        steps.append(("reduce_pre", DimensionalityReducerTransformer(
            method="pca", n_components=10)))
    # 6) Feature-Selection (optional)
    if fs_method:
        steps.append(("select", FeatureSelectionTransformer(
            method=fs_method, k=fs_k)))
    # 7) Post-DimRed
    if reduce_post_method == "pca":
        steps.append(("reduce_post", DimensionalityReducerTransformer(
            method="pca", n_components=10)))
    return steps


class PipelineWrapper:
    def __init__(
        self,
        model_keys: Optional[List[str]] = None,
        fs_method: Union[None, str] = "random_forest",  # default
        fs_k: Optional[int] = None,                     # None = all features
        impute_method: Optional[str] = None,
        outlier_method: Optional[str] = None,
        scaler_method: Optional[str] = None,
        reduce_pre_method: Optional[str] = None,
        reduce_post_method: Optional[str] = None,
        validation_strategy: str = "custom",            # "custom" or "cv5"
    ):
        RESULTS_DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.model_keys = model_keys or list(MODELS.keys())

        # Validate FS-method
        if fs_method not in (None, "all", *_ALLOWED_FS):
            raise ValueError(f"fs_method must be one of None, 'all', {_ALLOWED_FS}, got {fs_method!r}")
        self.fs_method         = fs_method
        self.fs_k              = fs_k
        self.impute_method     = impute_method
        self.outlier_method    = outlier_method
        self.scaler_method     = scaler_method
        self.reduce_pre_method = reduce_pre_method
        self.reduce_post_method= reduce_post_method

        if validation_strategy not in ("custom", "cv5"):
            raise ValueError("validation_strategy must be 'custom' or 'cv5'")
        self.validation_strategy = validation_strategy

        self._model_dir = Path(RESULTS_DATA_DIR) / "models"
        self._model_dir.mkdir(parents=True, exist_ok=True)

    def run(self) -> pd.DataFrame:
        # 1) Splits generieren, falls fehlen
        ensure_valid_splits()


        records: list[dict] = []

        # FS-method loop: einzelne oder alle
        fs_methods = []
        if self.fs_method == "all":
            fs_methods = _ALLOWED_FS.copy()
        elif self.fs_method:
            fs_methods = [self.fs_method]
        else:
            fs_methods = [None]

        # 2) Für jede FS-Variante und jede Species
        for fs_m in fs_methods:
            for species_dir in sorted(Path(SPLITS_DIR).iterdir()):
                if not species_dir.is_dir(): continue
                train_p = species_dir / "train.parquet"
                test_p  = species_dir / "test.parquet"
                if not train_p.exists() or not test_p.exists(): continue

                key = species_dir.name
                if key in _DATA_CACHE:
                    df_train = _DATA_CACHE[key]["train"]
                    df_test  = _DATA_CACHE[key]["test"]
                else:
                    df_train = (pd.read_parquet(train_p)
                                  .dropna(subset=["sex"])
                                  .query("sex in ['f','m']"))
                    df_test  = (pd.read_parquet(test_p)
                                  .dropna(subset=["sex"])
                                  .query("sex in ['f','m']"))
                    _DATA_CACHE[key] = {"train": df_train, "test": df_test}

                y_train = df_train["sex"].map({"f":0,"m":1})
                y_test  = df_test["sex"].map({"f":0,"m":1})

                # 3) Folds / CV
                if self.validation_strategy == "custom":
                    fold_ids, cv_method = _make_folds(
                        df_train, y_train, NUM_FOLDS, GROUP_COL
                    )
                    df_train = df_train.assign(Fold=fold_ids)
                    cv = PredefinedSplit(test_fold=fold_ids)
                else:
                    cv_method = "cv5"
                    kf = KFold(n_splits=NUM_FOLDS,
                               shuffle=True,
                               random_state=GLOBAL_RANDOM_SEED)
                    fold_ids = np.empty(len(df_train), dtype=int)
                    for fold, (_, idx) in enumerate(kf.split(df_train)):
                        fold_ids[idx] = fold
                    df_train = df_train.assign(Fold=fold_ids)
                    cv = PredefinedSplit(test_fold=fold_ids)

                X_train = df_train.drop(columns=["Fold"])
                X_test  = df_test

                # 4) Modelle
                for mk in self.model_keys:
                    model = MODELS[mk]
                    steps = get_pipeline_steps(
                        fs_method         = fs_m,
                        fs_k              = self.fs_k,
                        impute_method     = self.impute_method,
                        outlier_method    = self.outlier_method,
                        scaler_method     = self.scaler_method,
                        reduce_pre_method = self.reduce_pre_method,
                        reduce_post_method= self.reduce_post_method,
                    )
                    steps.append(("classifier", model))
                    pipe = Pipeline(steps, memory=memory)

                    # CV scores
                    try:
                        acc = cross_val_score(pipe, X_train, y_train,
                                              cv=cv, scoring="accuracy", n_jobs=1)
                        bal = cross_val_score(pipe, X_train, y_train,
                                              cv=cv, scoring="balanced_accuracy", n_jobs=1)
                        cv_acc_mean = float(acc.mean())
                        cv_bal_mean = float(bal.mean())
                    except Exception:
                        cv_acc_mean = None
                        cv_bal_mean = None

                    # Timing & fit
                    t_start = perf_counter()
                    X_ = X_train.copy()
                    times = {}
                    for name, step in pipe.steps[:-1]:
                        t0 = perf_counter()
                        X_ = (step.fit_transform(X_, y_train)
                              if hasattr(step, "fit_transform")
                              else step.fit(X_, y_train).transform(X_))
                        times[f"time_{name}"] = perf_counter() - t0
                    clf = pipe.steps[-1][1]
                    t0 = perf_counter()
                    clf.fit(X_, y_train)
                    times["time_classifier"] = perf_counter() - t0
                    t0 = perf_counter()
                    y_pred = pipe.predict(X_test)
                    times["time_predict"] = perf_counter() - t0
                    elapsed = perf_counter() - t_start

                    # Metrics
                    test_acc     = accuracy_score(y_test, y_pred)
                    test_bal_acc = balanced_accuracy_score(y_test, y_pred)
                    report       = classification_report(y_test, y_pred,
                                                         output_dict=True)

                    fs_trans       = pipe.named_steps.get("select")
                    selected_feats = getattr(fs_trans, "selected_features_", None)
                    ranking        = getattr(fs_trans, "feature_ranking_", None)

                    rec = {
                        "species":                key,
                        "model":                  mk,
                        "fs_method":              fs_m,
                        "fs_k":                   self.fs_k,
                        "cv_method":              cv_method,
                        "validation_strategy":    self.validation_strategy,
                        "cv_accuracy":            cv_acc_mean,
                        "cv_balanced_accuracy":   cv_bal_mean,
                        "test_accuracy":          float(test_acc),
                        "test_balanced_accuracy": float(test_bal_acc),
                        "time_total":             elapsed,
                        "classification_report":  report,
                        "selected_features":      selected_feats,
                        "feature_ranking":        ranking,
                        "impute_method":          self.impute_method,
                        "outlier_method":         self.outlier_method,
                        "scaler_method":          self.scaler_method,
                        "reduce_pre_method":      self.reduce_pre_method,
                        "reduce_post_method":     self.reduce_post_method,
                    }
                    rec.update(times)
                    records.append(rec)

        # 5) raw & best speichern
        df_new = pd.DataFrame(records)
        raw_out  = Path(RESULTS_DATA_DIR) / "raw_results.csv"
        best_out = Path(RESULTS_DATA_DIR) / "best_results.csv"

        df_raw = (pd.concat([pd.read_csv(raw_out), df_new], ignore_index=True)
                  if raw_out.exists() else df_new)
        df_raw.to_csv(raw_out, index=False)

        df_comb = (pd.concat([pd.read_csv(best_out), df_new], ignore_index=True)
                   if best_out.exists() else df_new)
        df_best = (
            df_comb
            .sort_values("test_balanced_accuracy", ascending=False)
            .drop_duplicates(subset=["species"], keep="first")
            .reset_index(drop=True)
        )
        df_best.to_csv(best_out, index=False)

        # 6) Finale Pipelines speichern
        for _, row in df_best.iterrows():
            species = row["species"]
            mk      = row["model"]
            df_t    = _DATA_CACHE[species]["train"]
            y_t     = df_t["sex"].map({"f":0,"m":1})
            X_t     = df_t.drop(columns=["Fold","sex"])
            steps = get_pipeline_steps(
                fs_method         = row["fs_method"],
                fs_k              = row["fs_k"],
                impute_method     = row["impute_method"],
                outlier_method    = row["outlier_method"],
                scaler_method     = row["scaler_method"],
                reduce_pre_method = row["reduce_pre_method"],
                reduce_post_method= row["reduce_post_method"],
            )
            steps.append(("classifier", MODELS[mk]))
            final_pipe = Pipeline(steps)
            final_pipe.fit(X_t, y_t)
            dump(final_pipe, self._model_dir / f"{species}.joblib")

        return df_raw
