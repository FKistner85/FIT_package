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

# Cache-Verzeichnis für sklearn-Pipelines
_cache_dir = Path(RESULTS_DATA_DIR) / "pipeline_cache"
memory = Memory(location=_cache_dir, verbose=0)

# In-Memory-Cache für geladene Splits
_DATA_CACHE: dict[str, dict[str, pd.DataFrame]] = {}

def splits_available() -> bool:
    """Prüft, ob mindestens ein gültiges Train/Test-Paar existiert."""
    if not Path(SPLITS_DIR).exists():
        return False
    for d in Path(SPLITS_DIR).iterdir():
        if not d.is_dir():
            continue
        if (d / "train.parquet").exists() and (d / "test.parquet").exists():
            return True
    return False

def get_pipeline_steps(
    fs_method: str = "forward",
    fs_k: int = 20,
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
    # 6) Feature-Selection
    steps.append(("select", FeatureSelectionTransformer(method=fs_method, k=fs_k)))
    # 7) DimRed nach Selektion
    if reduce_post_method == "pca":
        steps.append(("reduce_post", DimensionalityReducerTransformer(method="pca", n_components=10)))
    return steps

class PipelineWrapper:
    def __init__(
        self,
        model_keys: list[str] | None = None,
        fs_method: str = "forward",
        fs_k: int = 20,
        impute_method: str | None = None,
        outlier_method: str | None = None,
        scaler_method: str | None = None,
        reduce_pre_method: str | None = None,
        reduce_post_method: str | None = None,
    ):
        RESULTS_DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.model_keys = model_keys or list(MODELS.keys())
        self.fs_method = fs_method
        self.fs_k = fs_k
        self.impute_method = impute_method
        self.outlier_method = outlier_method
        self.scaler_method = scaler_method
        self.reduce_pre_method = reduce_pre_method
        self.reduce_post_method = reduce_post_method
        # Verzeichnis für gespeicherte Modelle
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

            # 5) PredefinedSplit via 'Fold'
            ps = PredefinedSplit(test_fold=df_train["Fold"].values)
            X_train = df_train.drop(columns=["Fold"])
            X_test  = df_test

            # 6) Loop über Modelle
            for mk in self.model_keys:
                model = MODELS[mk]
                # a) Pipeline zusammenbauen
                steps = get_pipeline_steps(
                    fs_method=self.fs_method,
                    fs_k=self.fs_k,
                    impute_method=self.impute_method,
                    outlier_method=self.outlier_method,
                    scaler_method=self.scaler_method,
                    reduce_pre_method=self.reduce_pre_method,
                    reduce_post_method=self.reduce_post_method,
                )
                steps.append(("classifier", model))
                pipe = Pipeline(steps, memory=memory)

                # b) Cross-Validation auf Trainingsdaten
                try:
                    acc_scores = cross_val_score(
                        pipe, X_train, y_train, cv=ps,
                        scoring="accuracy", n_jobs=1
                    )
                    bal_scores = cross_val_score(
                        pipe, X_train, y_train, cv=ps,
                        scoring="balanced_accuracy", n_jobs=1
                    )
                    cv_acc_mean = float(acc_scores.mean())
                    cv_bal_mean = float(bal_scores.mean())
                except Exception:
                    cv_acc_mean = None
                    cv_bal_mean = None

                # c) Gesamtzeit messen
                t_start = perf_counter()

                # d) Finales Fit + Timing je Schritt
                times: dict[str, float] = {}
                X = X_train.copy()
                for name, step in pipe.steps[:-1]:
                    t0 = perf_counter()
                    X = step.fit_transform(X, y_train) if hasattr(step, "fit_transform") \
                        else step.fit(X, y_train).transform(X)
                    times[f"time_{name}"] = perf_counter() - t0

                clf = pipe.steps[-1][1]
                t0 = perf_counter()
                clf.fit(X, y_train)
                times["time_classifier"] = perf_counter() - t0

                # e) Vorhersage auf Test + Timing
                t0 = perf_counter()
                y_pred = pipe.predict(X_test)
                times["time_predict"] = perf_counter() - t0

                time_total = perf_counter() - t_start

                # f) Test-Accuracy und Balanced-Accuracy
                test_acc     = float(accuracy_score(y_test, y_pred))
                test_bal_acc = float(balanced_accuracy_score(y_test, y_pred))
                report       = classification_report(y_test, y_pred, output_dict=True)

                # g) Feature-Selection-Info
                fs_trans       = pipe.named_steps["select"]
                selected_feats = fs_trans.selected_features_
                ranking        = fs_trans.feature_ranking_

                # h) Record anlegen
                rec = {
                    "species":                key,
                    "model":                  mk,
                    "cv_accuracy":            cv_acc_mean,
                    "cv_balanced_accuracy":   cv_bal_mean,
                    "test_accuracy":          test_acc,
                    "test_balanced_accuracy": test_bal_acc,
                    "classification_report":  report,
                    "fs_method":              self.fs_method,
                    "fs_k":                   self.fs_k,
                    "selected_features":      selected_feats,
                    "feature_ranking":        ranking,
                    "impute_method":          self.impute_method,
                    "outlier_method":         self.outlier_method,
                    "scaler_method":          self.scaler_method,
                    "reduce_pre_method":      self.reduce_pre_method,
                    "reduce_post_method":     self.reduce_post_method,
                    "time_total":             time_total,
                }
                rec.update(times)
                records.append(rec)

        # 7) Alle Runs zu DataFrame
        df_new = pd.DataFrame(records)

        # 8) Raw-Results anhängen
        raw_out = Path(RESULTS_DATA_DIR) / "raw_results.csv"
        if raw_out.exists():
            df_raw = pd.concat([pd.read_csv(raw_out), df_new], ignore_index=True)
        else:
            df_raw = df_new.copy()
        df_raw.to_csv(raw_out, index=False)

        # 9) Best-Results pro Spezies aktualisieren
        best_out = Path(RESULTS_DATA_DIR) / "best_results.csv"
        if best_out.exists():
            df_comb = pd.concat([pd.read_csv(best_out), df_new], ignore_index=True)
        else:
            df_comb = df_new.copy()

        df_best = (
            df_comb
            .sort_values("test_balanced_accuracy", ascending=False)
            .drop_duplicates(subset=["species"], keep="first")
            .reset_index(drop=True)
        )
        df_best.to_csv(best_out, index=False)

        # 10) Finale Modelle speichern (vollständige Pipeline)
        for _, row in df_best.iterrows():
            species = row["species"]
            mk      = row["model"]
            df_t    = _DATA_CACHE[species]["train"]
            y_t     = df_t["sex"].map({"f": 0, "m": 1})
            X_t     = df_t.drop(columns=["Fold", "sex"])

            steps = get_pipeline_steps(
                fs_method=row["fs_method"],
                fs_k=row["fs_k"],
                impute_method=row["impute_method"],
                outlier_method=row["outlier_method"],
                scaler_method=row["scaler_method"],
                reduce_pre_method=row["reduce_pre_method"],
                reduce_post_method=row["reduce_post_method"],
            )
            steps.append(("classifier", MODELS[mk]))
            final_pipe = Pipeline(steps)
            final_pipe.fit(X_t, y_t)

            dump(final_pipe, self._model_dir / f"{species}.joblib")

        # 11) Return aller Roh-Ergebnisse
        return df_raw
