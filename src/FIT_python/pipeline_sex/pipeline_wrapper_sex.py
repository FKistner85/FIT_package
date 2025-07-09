# src/FIT_python/pipeline/pipeline_wrapper_sex.py

from pathlib import Path
import pandas as pd
import numpy as np
from typing import Optional, Union, List
from joblib import Memory, dump
from time import perf_counter
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.pipeline import Pipeline
from sklearn.metrics import balanced_accuracy_score, classification_report
from sklearn.model_selection import cross_val_score, cross_val_predict, PredefinedSplit

from FIT_python.config import (
    SPLITS_DIR,
    RESULTS_DATA_DIR,
    FIGURES_DIR,
    GLOBAL_RANDOM_SEED,
)

# Utility functions
from FIT_python.data_split_and_summary.split_utils import (
    ensure_valid_splits,
)

# Wrappers for data preparation
from FIT_python.data_split_and_summary.data_import_wrapper import DataImportWrapper
from FIT_python.data_split_and_summary.split_wrapper import SplitWrapper
from FIT_python.data_split_and_summary.summary_data_wrapper import SummaryWrapper

# Wrappers for pipeline steps
from FIT_python.pipeline_sex.transform_wrapper import NumericTransformer
from FIT_python.pipeline_sex.imputation_wrapper import ImputationWrapper
from FIT_python.pipeline_sex.outlier_wrapper import OutlierCleanerTransformer
from FIT_python.pipeline_sex.feature_scaler_wrapper import FeatureScalerTransformer
from FIT_python.pipeline_sex.feature_selection_wrapper import (
    FeatureSelectionTransformer,
)
from FIT_python.pipeline_sex.dimensionality_reduction_wrapper import (
    DimensionalityReducerTransformer,
)
from FIT_python.pipeline_sex.models import MODELS
from FIT_python.pipeline_sex.sex_predict_and_visualisation import (
    plot_hyperparam_heatmap,
)

# Cache for sklearn Pipelines
_cache_dir = Path(RESULTS_DATA_DIR) / "pipeline_cache"
memory = Memory(location=_cache_dir, verbose=0)

# In-memory cache of loaded splits
_DATA_CACHE: dict[str, dict[str, pd.DataFrame]] = {}

# Allowed hyperparameter values
_ALLOWED_FS = [None, "forward", "random_forest", "variance", "univariate", "lasso"]
_ALLOWED_IMPUTE = [None, "miss_forest"]
_ALLOWED_OUTLIERS = [None, "clip", "zscore"]
_ALLOWED_SCALERS = [None, "standard", "robust"]
_ALLOWED_REDS = [None, "pca", "umap", "tsne"]


def get_pipeline_steps(
    fs_method: Optional[str] = None,
    fs_k: Optional[int] = None,
    impute_method: Optional[str] = None,
    outlier_method: Optional[str] = None,
    scaler_method: Optional[str] = None,
    reduce_pre_method: Optional[str] = None,
    reduce_post_method: Optional[str] = None,
) -> list[tuple[str, object]]:
    """Construct the list of ``(name, transformer)`` steps based on the chosen hyperparameters."""
    if fs_method not in _ALLOWED_FS:
        raise ValueError(f"fs_method must be one of {_ALLOWED_FS}, got {fs_method!r}")
    if impute_method not in _ALLOWED_IMPUTE:
        raise ValueError(
            f"impute_method must be one of {_ALLOWED_IMPUTE}, got {impute_method!r}"
        )
    if outlier_method not in _ALLOWED_OUTLIERS:
        raise ValueError(
            f"outlier_method must be one of {_ALLOWED_OUTLIERS}, got {outlier_method!r}"
        )
    if scaler_method not in _ALLOWED_SCALERS:
        raise ValueError(
            f"scaler_method must be one of {_ALLOWED_SCALERS}, got {scaler_method!r}"
        )
    if reduce_pre_method not in _ALLOWED_REDS:
        raise ValueError(
            f"reduce_pre_method must be one of {_ALLOWED_REDS}, got {reduce_pre_method!r}"
        )
    if reduce_post_method not in _ALLOWED_REDS:
        raise ValueError(
            f"reduce_post_method must be one of {_ALLOWED_REDS}, got {reduce_post_method!r}"
        )

    steps: list[tuple[str, object]] = []

    # 2) Numeric conversion
    steps.append(("transform", NumericTransformer()))

    # 3) Imputation
    if impute_method == "miss_forest":
        steps.append(("impute", ImputationWrapper()))

    # 4) Outlier cleaning
    if outlier_method == "clip":
        steps.append(
            (
                "outlier",
                OutlierCleanerTransformer(
                    method="clip", lower_quantile=0.01, upper_quantile=0.99
                ),
            )
        )
    elif outlier_method == "zscore":
        steps.append(
            ("outlier", OutlierCleanerTransformer(method="zscore", z_thresh=3.0))
        )

    # 5) Scaling
    if scaler_method in ("standard", "robust"):
        steps.append(("scale", FeatureScalerTransformer(method=scaler_method)))

    # 6) Pre-dimensionality reduction
    if reduce_pre_method in ("pca", "umap", "tsne"):
        steps.append(
            (
                "reduce_pre",
                DimensionalityReducerTransformer(
                    method=reduce_pre_method, n_components=10
                ),
            )
        )

    # 7) Feature selection
    if fs_method:
        steps.append(("select", FeatureSelectionTransformer(method=fs_method, k=fs_k)))

    # 8) Post-dimensionality reduction
    if reduce_post_method in ("pca", "umap", "tsne"):
        steps.append(
            (
                "reduce_post",
                DimensionalityReducerTransformer(
                    method=reduce_post_method, n_components=10
                ),
            )
        )

    return steps


class PipelineWrapper:
    """Wraps one-time preparation (import, split, summary) and
    training/evaluation of all model variants."""

    def __init__(
        self,
        model_keys: Optional[List[str]] = None,
        fs_method: Union[None, str] = "random_forest",
        fs_k: Optional[int] = None,
        impute_method: Optional[str] = None,
        outlier_method: Optional[str] = None,
        scaler_method: Optional[str] = None,
        reduce_pre_method: Optional[str] = None,
        reduce_post_method: Optional[str] = None,
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

        self._model_dir = Path(RESULTS_DATA_DIR) / "sex_models"
        self._model_dir.mkdir(parents=True, exist_ok=True)
        self._best_dir = Path(RESULTS_DATA_DIR) / "sex_models_best"
        self._best_dir.mkdir(parents=True, exist_ok=True)

    def prepare(self):
        """Run data import, splitting and summary exactly once."""
        print("\n📥 Schritt 1: Datenimport & Cleaning")
        DataImportWrapper().clean_all()
        print("\n✂️ Schritt 2: Splitting & Fold-Zuordnung")
        SplitWrapper().split_all()
        print("\n📊 Schritt 3: Zusammenfassung der Splits")
        SummaryWrapper().summarize_all()
        ensure_valid_splits()

    def train(self) -> pd.DataFrame:
        """Train all models, write ``raw_results.csv`` and store all
        fitted pipelines as well as the best per species."""
        records: list[dict] = []
        best_acc_per_species: dict[str, float] = {}

        # iterate over all feature-selection variants
        fs_methods = [self.fs_method] if self.fs_method else [None]
        for fs_m in fs_methods:
            for species_dir in sorted(Path(SPLITS_DIR).iterdir()):
                if not species_dir.is_dir():
                    continue

                key = species_dir.name
                best_acc_per_species.setdefault(key, -np.inf)

                train_fp = species_dir / "train.parquet"
                test_fp = species_dir / "test.parquet"
                if not train_fp.exists() or not test_fp.exists():
                    continue

                # load data
                if key in _DATA_CACHE:
                    df_train = _DATA_CACHE[key]["train"]
                    df_test = _DATA_CACHE[key]["test"]
                else:
                    df_train = (
                        pd.read_parquet(train_fp)
                        .dropna(subset=["sex"])
                        .query("sex in ['f','m']")
                    )
                    df_test = (
                        pd.read_parquet(test_fp)
                        .dropna(subset=["sex"])
                        .query("sex in ['f','m']")
                    )
                    _DATA_CACHE[key] = {"train": df_train, "test": df_test}

                y_train = df_train["sex"].map({"f": 0, "m": 1})
                y_test = df_test["sex"].map({"f": 0, "m": 1})

                drop_pred_train = [c for c in df_train.columns if c.startswith("pred_")]
                drop_pred_test = [c for c in df_test.columns if c.startswith("pred_")]

                if "Fold" in df_train.columns:
                    fold_ids = df_train["Fold"].astype(int).to_numpy()
                    X_train = df_train.drop(columns=["Fold", *drop_pred_train])
                    cv = PredefinedSplit(test_fold=fold_ids)
                else:
                    X_train = df_train.drop(columns=drop_pred_train)
                    cv = 5

                if "Fold" in df_test.columns:
                    X_test = df_test.drop(columns=["Fold", *drop_pred_test])
                else:
                    X_test = df_test.drop(columns=drop_pred_test)

                # iterate over all models
                for mk in self.model_keys:
                    model = MODELS[mk]
                    steps = get_pipeline_steps(
                        fs_method=fs_m,
                        fs_k=self.fs_k,
                        impute_method=self.impute_method,
                        outlier_method=self.outlier_method,
                        scaler_method=self.scaler_method,
                        reduce_pre_method=self.reduce_pre_method,
                        reduce_post_method=self.reduce_post_method,
                    )
                    steps.append(("classifier", model))
                    pipe = Pipeline(steps, memory=memory)

                    # cross-val using balanced accuracy and out-of-fold predictions
                    try:
                        bal = cross_val_score(
                            pipe,
                            X_train,
                            y_train,
                            cv=cv,
                            scoring="balanced_accuracy",
                            n_jobs=1,
                        )
                        y_pred_cv = cross_val_predict(
                            pipe,
                            X_train,
                            y_train,
                            cv=cv,
                            n_jobs=1,
                        )
                        cv_bal_mean = float(bal.mean())
                    except Exception:
                        cv_bal_mean = None
                        y_pred_cv = np.full(len(y_train), np.nan)
                    # store oof predictions
                    df_train[f"pred_{mk}_cv_sex"] = y_pred_cv

                    # fit & predict
                    t_start = perf_counter()
                    X_tmp = X_train.copy()
                    times = {}
                    for name, step in pipe.steps[:-1]:
                        t0 = perf_counter()
                        X_tmp = (
                            step.fit_transform(X_tmp, y_train)
                            if hasattr(step, "fit_transform")
                            else step.fit(X_tmp, y_train).transform(X_tmp)
                        )
                        times[f"time_{name}"] = perf_counter() - t0

                    clf = pipe.steps[-1][1]
                    t0 = perf_counter()
                    clf.fit(X_tmp, y_train)
                    times["time_classifier"] = perf_counter() - t0
                    t0 = perf_counter()
                    y_pred = pipe.predict(X_test)
                    times["time_predict"] = perf_counter() - t0

                    # metrics
                    test_bal_acc = balanced_accuracy_score(y_test, y_pred)
                    report = classification_report(y_test, y_pred, output_dict=True)
                    fs_trans = pipe.named_steps.get("select")
                    selected = getattr(fs_trans, "selected_features_", None)
                    ranking = getattr(fs_trans, "feature_ranking_", None)

                    # record
                    rec = {
                        "species": key,
                        "model": mk,
                        "fs_method": fs_m,
                        "fs_k": self.fs_k,
                        "cv_balanced_accuracy": cv_bal_mean,
                        "test_balanced_accuracy": float(test_bal_acc),
                        "classification_report": report,
                        "selected_features": selected,
                        "feature_ranking": ranking,
                        "impute_method": self.impute_method,
                        "outlier_method": self.outlier_method,
                        "scaler_method": self.scaler_method,
                        "reduce_pre_method": self.reduce_pre_method,
                        "reduce_post_method": self.reduce_post_method,
                        **times,
                        "time_total": perf_counter() - t_start,
                    }
                    records.append(rec)

                # end for mk

                # persist CV predictions for this species
                df_train.to_parquet(train_fp, index=False)
                df_train.to_csv(species_dir / "train.csv", index=False)
                _DATA_CACHE[key]["train"] = df_train

        # save raw_results.csv
        df_new = pd.DataFrame(records)
        raw_out = Path(RESULTS_DATA_DIR) / "raw_results.csv"
        df_new.to_csv(raw_out, mode="a", header=not raw_out.exists(), index=False)

        # aggregate timing columns
        time_cols = [c for c in df_new.columns if c.startswith("time_")]
        if time_cols:
            time_df = (
                df_new[time_cols].mean().rename_axis("step").reset_index(name="seconds")
            )
            time_df["step"] = time_df["step"].str.replace("time_", "", regex=False)
            plot_pipeline_timings(time_df, Path(FIGURES_DIR) / "pipeline_timings")

        # summarise cross-validation scores by preprocessing options
        self.pivot_cv = (
            df_new.pivot_table(
                index="fs_method",
                columns="reduce_pre_method",
                values="cv_balanced_accuracy",
                aggfunc="mean",
            )
        )

        # Visualise the hyperparameter search results
        plot_hyperparam_heatmap(
            df_new, Path(FIGURES_DIR) / "hyperparam_search"
        )

        # finale pipelines fit & dump
        for _, row in df_new.iterrows():
            species = row["species"]
            mk = row["model"]
            df_t = _DATA_CACHE[species]["train"]
            y_t = df_t["sex"].map({"f": 0, "m": 1})
            X_t = df_t.drop(columns=["Fold", "sex"])
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

            # 1) Save every model using a filename that encodes the hyperparameters
            fname_all = (
                f"{species}__{mk}"
                f"__fs-{row['fs_method'] or 'none'}-{row['fs_k']}"
                f"__impute-{row['impute_method'] or 'none'}"
                f"__outlier-{row['outlier_method'] or 'none'}"
                f"__scaler-{row['scaler_method'] or 'none'}"
                f"__redpre-{row['reduce_pre_method'] or 'none'}"
                f"__redpost-{row['reduce_post_method'] or 'none'}.joblib"
            )
            dump(final_pipe, self._model_dir / fname_all)

            # 2) Best model per species
            best_path = self._best_dir / f"{species}.joblib"
            if row["test_balanced_accuracy"] >= best_acc_per_species[species]:
                dump(final_pipe, best_path)
                best_acc_per_species[species] = row["test_balanced_accuracy"]

        return df_new



def plot_pipeline_timings(time_df: pd.DataFrame, out_dir: Path) -> None:
    """Create a bar chart of average seconds per preprocessing step."""
    from FIT_python.plot_style import apply_style
    import matplotlib.pyplot as plt

    apply_style()
    out_dir.mkdir(parents=True, exist_ok=True)

    order = time_df.sort_values("seconds", ascending=False)

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.bar(order["step"], order["seconds"], color="#4C72B0", edgecolor="black")
    ax.set_xlabel("Step")
    ax.set_ylabel("Average Seconds")
    plt.xticks(rotation=45, ha="right")
    fig.tight_layout()

    from FIT_python.caption_utils import save_caption

    caption = "Average seconds per preprocessing step"
    for ext in ("png", "svg"):
        out_file = out_dir / f"pipeline_timings.{ext}"
        fig.savefig(out_file)
        save_caption(out_file, caption)
    plt.close(fig)


