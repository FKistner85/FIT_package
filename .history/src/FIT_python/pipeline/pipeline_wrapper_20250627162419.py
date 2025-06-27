# src/FIT_python/pipeline/pipeline_wrapper.py

from pathlib import Path
import pandas as pd
import numpy as np
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
    StratifiedGroupKFold,
    GroupKFold,
    KFold,
)

from FIT_python.config import (
    SPLITS_DIR,
    RESULTS_DATA_DIR,
    GLOBAL_RANDOM_SEED,
    NUM_FOLDS,
    GROUP_COL,
)
from FIT_python.step_02_a_splitting_train_test.wrapper import SplitWrapper
from .transform_wrapper import NumericTransformer
from .imputation_wrapper import ImputationWrapper
from .outlier_wrapper import OutlierCleanerTransformer
from .feature_scaler_wrapper import FeatureScalerTransformer
from .feature_selection_wrapper import FeatureSelectionTransformer
from .dimensionality_reduction_wrapper import DimensionalityReducerTransformer
from .models import MODELS

# Cache-Verzeichnis für sklearn-Pipelines
_cache_dir = Path(RESULTS_DATA_DIR) / "pipeline_cache"
memory = Memory(location=_cache_dir, verbose=0)

# In-Memory-Cache für geladene Splits
_DATA_CACHE: dict[str, dict[str, pd.DataFrame]] = {}



def get_pipeline_steps(
    fs_method: str | None = None,
    fs_k: int | None = None,
    impute_method: str | None = None,
    outlier_method: str | None = None,
    scaler_method: str | None = None,
    reduce_pre_method: str | None = None,
    reduce_post_method: str | None = None
) -> list[tuple[str, object]]:
    """Stellt die Liste der Pipeline-Schritte anhand der Parameter zusammen."""
    steps: list[tuple[str, object]] = []
    # 1) Roh-Transformation
    steps.append(("transform", NumericTransformer()))
    # 2) Imputation
    if impute_method == "miss_forest":
        steps.append(("impute", ImputationWrapper()))
    # 3) Outlier-Bereinigung
    if outlier_method == "clip":
        steps.append((
            "outlier",
            OutlierCleanerTransformer(method="clip", lower_quantile=0.01, upper_quantile=0.99)
        ))
    # 4) Skalierung
    if scaler_method in ("standard", "robust"):
        steps.append(("scale", FeatureScalerTransformer(method=scaler_method)))
    # 5) DimRed vor Selektion
    if reduce_pre_method == "pca":
        steps.append(("reduce_pre", DimensionalityReducerTransformer(method="pca", n_components=10)))
    # 6) Feature-Selection (optional)
    if fs_method:
        steps.append(("select", FeatureSelectionTransformer(method=fs_method, k=fs_k)))
    # 7) DimRed nach Selektion
    if reduce_post_method == "pca":
        steps.append(("reduce_post", DimensionalityReducerTransformer(method="pca", n_components=10)))
    return steps

class PipelineWrapper:
    def __init__(
        self,
        model_keys: list[str] | None = None,
        fs_method: str | None = None,      # None = keine Selektion
        fs_k: int | None = None,           # None = alle Features
        impute_method: str | None = None,
        outlier_method: str | None = None,
        scaler_method: str | None = None,
        reduce_pre_method: str | None = None,
        reduce_post_method: str | None = None,
        validation_strategy: str = "custom",  # "custom" oder "cv5"
    ):
        # Verzeichnis für alle Resultate anlegen
        RESULTS_DATA_DIR.mkdir(parents=True, exist_ok=True)

        # Modelle auswählen (None → alle aus MODELS)
        self.model_keys = model_keys or list(MODELS.keys())

        # Parameter speichern
        self.fs_method            = fs_method
        self.fs_k                 = fs_k
        self.impute_method        = impute_method
        self.outlier_method       = outlier_method
        self.scaler_method        = scaler_method
        self.reduce_pre_method    = reduce_pre_method
        self.reduce_post_method   = reduce_post_method

        # Validierungsstrategie
        if validation_strategy not in ("custom", "cv5"):
            raise ValueError("validation_strategy must be 'custom' or 'cv5'")
        self.validation_strategy = validation_strategy

        # Verzeichnis für finale (beste) Modelle
        self._model_dir = Path(RESULTS_DATA_DIR) / "models"
        self._model_dir.mkdir(parents=True, exist_ok=True)

    def run(self) -> pd.DataFrame:
        # 1) Splits erzeugen, falls nötig
        if not splits_available():
            SplitWrapper().split_all()

        records: list[dict] = []

        # 2) Pro Spezies-Ordner
        for species_dir in sorted(Path(SPLITS_DIR).iterdir()):
            if not species_dir.is_dir():
                continue
            train_p = species_dir / "train.parquet"
            test_p  = species_dir / "test.parquet"
            if not train_p.exists() or not test_p.exists():
                continue

            # 3) DataFrame-Caching
            key = species_dir.name
            if key in _DATA_CACHE:
                df_train = _DATA_CACHE[key]["train"]
                df_test  = _DATA_CACHE[key]["test"]
            else:
                df_train = (
                    pd.read_parquet(train_p)
                      .dropna(subset=["sex"])
                      .query("sex in ['f','m']")
                )
                df_test = (
                    pd.read_parquet(test_p)
                      .dropna(subset=["sex"])
                      .query("sex in ['f','m']")
                )
                _DATA_CACHE[key] = {"train": df_train, "test": df_test}

            # 4) Label-Encoding
            y_train = df_train["sex"].map({"f": 0, "m": 1})
            y_test  = df_test["sex"].map({"f": 0, "m": 1})

            # 5) Folds & CV-Objekt je nach Strategie
            if self.validation_strategy == "custom":
                fold_ids, cv_method = _make_folds(
                    df_train, y_train, NUM_FOLDS, GROUP_COL
                )
                df_train = df_train.assign(Fold=fold_ids)
                cv = PredefinedSplit(test_fold=fold_ids)
            else:  # cv5
                cv_method = "cv5"
                kf = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=GLOBAL_RANDOM_SEED)
                fold_ids = np.empty(len(df_train), dtype=int)
                for fold, (_, val_idx) in enumerate(kf.split(df_train)):
                    fold_ids[val_idx] = fold
                df_train = df_train.assign(Fold=fold_ids)
                cv = PredefinedSplit(test_fold=fold_ids)

            # 6) Trainings- und Test‐Matrizen
            X_train = df_train.drop(columns=["Fold"])
            X_test  = df_test

            # 7) Loop über Modelle
            for mk in self.model_keys:
                model = MODELS[mk]

                # a) Pipeline zusammenbauen
                steps = get_pipeline_steps(
                    fs_method        = self.fs_method,
                    fs_k             = self.fs_k,
                    impute_method    = self.impute_method,
                    outlier_method   = self.outlier_method,
                    scaler_method    = self.scaler_method,
                    reduce_pre_method = self.reduce_pre_method,
                    reduce_post_method= self.reduce_post_method,
                )
                steps.append(("classifier", model))
                pipe = Pipeline(steps, memory=memory)

                # b) Cross-Validation (Accuracy + Balanced)
                try:
                    acc_scores = cross_val_score(
                        pipe, X_train, y_train, cv=cv,
                        scoring="accuracy", n_jobs=1
                    )
                    bal_scores = cross_val_score(
                        pipe, X_train, y_train, cv=cv,
                        scoring="balanced_accuracy", n_jobs=1
                    )
                    cv_acc_mean = float(acc_scores.mean())
                    cv_bal_mean = float(bal_scores.mean())
                except Exception:
                    cv_acc_mean = None
                    cv_bal_mean = None

                # c) Gesamtzeit messen
                t_start = perf_counter()

                # d) Fit + Timing pro Schritt
                times: dict[str, float] = {}
                X = X_train.copy()
                for name, step in pipe.steps[:-1]:
                    t0 = perf_counter()
                    X = step.fit_transform(X, y_train) \
                        if hasattr(step, "fit_transform") \
                        else step.fit(X, y_train).transform(X)
                    times[f"time_{name}"] = perf_counter() - t0

                # Klassifikator
                clf = pipe.steps[-1][1]
                t0 = perf_counter()
                clf.fit(X, y_train)
                times["time_classifier"] = perf_counter() - t0

                # e) Vorhersage + Timing
                t0 = perf_counter()
                y_pred = pipe.predict(X_test)
                times["time_predict"] = perf_counter() - t0

                time_total = perf_counter() - t_start

                # f) Test-Accuracy & Balanced-Accuracy
                test_acc     = accuracy_score(y_test, y_pred)
                test_bal_acc = balanced_accuracy_score(y_test, y_pred)
                report       = classification_report(y_test, y_pred, output_dict=True)

                # g) Feature‐Selection‐Info
                fs_trans       = pipe.named_steps.get("select", None)
                selected_feats = getattr(fs_trans, "selected_features_", None)
                ranking        = getattr(fs_trans, "feature_ranking_", None)

                # h) Record zusammenbauen
                rec = {
                    "species":                 key,
                    "model":                   mk,
                    "cv_method":               cv_method,
                    "validation_strategy":     self.validation_strategy,
                    "cv_accuracy":             cv_acc_mean,
                    "cv_balanced_accuracy":    cv_bal_mean,
                    "test_accuracy":           float(test_acc),
                    "test_balanced_accuracy":  float(test_bal_acc),
                    "time_total":              time_total,
                    "classification_report":   report,
                    "fs_method":               self.fs_method,
                    "fs_k":                    self.fs_k,
                    "selected_features":       selected_feats,
                    "feature_ranking":         ranking,
                    "impute_method":           self.impute_method,
                    "outlier_method":          self.outlier_method,
                    "scaler_method":           self.scaler_method,
                    "reduce_pre_method":       self.reduce_pre_method,
                    "reduce_post_method":      self.reduce_post_method,
                }
                rec.update(times)
                records.append(rec)

        # 8) Alle Runs zu DataFrame
        df_new = pd.DataFrame(records)

        # 9) Raw-Results anhängen
        raw_out = Path(RESULTS_DATA_DIR) / "raw_results.csv"
        if raw_out.exists():
            df_raw = pd.concat([pd.read_csv(raw_out), df_new], ignore_index=True)
        else:
            df_raw = df_new.copy()
        df_raw.to_csv(raw_out, index=False)

        # 10) Best-Results pro Spezies aktualisieren
        best_out = Path(RESULTS_DATA_DIR) / "best_results.csv"
        if best_out.exists():
            df_comb = pd.concat(
                [pd.read_csv(best_out), df_new],
                ignore_index=True
            )
        else:
            df_comb = df_new.copy()

        df_best = (
            df_comb
            .sort_values("test_balanced_accuracy", ascending=False)
            .drop_duplicates(subset=["species"], keep="first")
            .reset_index(drop=True)
        )
        df_best.to_csv(best_out, index=False)

        # 11) Finale Modelle speichern (nur Train-Split)
        for _, row in df_best.iterrows():
            species = row["species"]
            mk      = row["model"]
            df_t    = _DATA_CACHE[species]["train"]
            y_t     = df_t["sex"].map({"f": 0, "m": 1})
            X_t     = df_t.drop(columns=["Fold", "sex"])

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

        # 12) Return aller Roh-Ergebnisse
        return df_raw
