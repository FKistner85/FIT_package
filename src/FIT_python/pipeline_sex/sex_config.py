"""Configuration and search utilities for sex classification pipelines."""

from __future__ import annotations
import pandas as pd
import numpy as np
import warnings
import joblib
from functools import reduce
import operator
from sklearn.exceptions import ConvergenceWarning
from collections import Counter
from sklearn.pipeline import Pipeline
from sklearn.base import clone
from skopt import BayesSearchCV
from skopt.space import Categorical
from sklearn.model_selection import PredefinedSplit
from tqdm.auto import tqdm
from tqdm_joblib import tqdm_joblib
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
    

from FIT_python.data_split_and_summary.transform_wrapper import NumericTransformer
from FIT_python.general_pipeline_steps.feature_selection_wrapper import (
    FeatureSelectionTransformer,
)
from FIT_python.general_pipeline_steps.outlier_wrapper import OutlierCleanerTransformer
from FIT_python.general_pipeline_steps.feature_scaler_wrapper import (
    FeatureScalerTransformer,
)
from FIT_python.general_pipeline_steps.dimensionality_reduction_wrapper import (
    DimensionalityReducerTransformer,
)
from FIT_python.general_pipeline_steps.models import MODELS
from FIT_python.soft_config import SOFT_CONFIG
from FIT_python.pipeline_sex.grouped_metrics import (
    individual_accuracies,
    individual_majority_stats,
)
from FIT_python.pipeline_sex.sex_predict_and_visualisation import (
    plot_hyperparam_heatmap,
    predict_all,
)


class EstimatorWrapper:
    """Simple container for an estimator without ``__len__``/``__iter__``.

    The wrapper proxies all estimator methods/attributes so it can be used
    transparently inside a :class:`~sklearn.pipeline.Pipeline` while ensuring
    that optimization libraries treat it as an atomic object.
    """

    def __init__(self, estimator):
        self.estimator = estimator

    def __getattr__(self, name):  # proxy to underlying estimator
        return getattr(self.estimator, name)

    def get_params(self, deep=True):
        # Only expose the wrapped estimator so ``sklearn.clone`` works.
        return {"estimator": self.estimator}

    def set_params(self, **params):
        if "estimator" in params:
            self.estimator = params.pop("estimator")
        if params:
            self.estimator.set_params(**params)
        return self

    def __getstate__(self):
        return {"estimator": self.estimator}

    def __setstate__(self, state):
        self.estimator = state["estimator"]

from FIT_python.data_split_and_summary.data_import_wrapper import DataImporter
from FIT_python.data_split_and_summary.data_import_utils import get_feature_cols
from FIT_python.data_split_and_summary.split_utils import (
    create_train_test_split_otter,
    _make_folds,
)
from FIT_python.data_split_and_summary.summary_data_wrapper import run_summary
from FIT_python.config import (
    RAW_DIR,
    SPLITS_DIR,
    RESULTS_DATA_DIR,
    DEFAULT_TARGETS,
    GROUP_COL,
    NUM_FOLDS,
    BAYES_REFIT,
    SEX_PREDICT_METRIC,
)
from FIT_python.utils import get_species_paths

PIPE_CFG = SOFT_CONFIG["pipeline_sex"]

MODEL_KEYS = PIPE_CFG["model_keys"]

SEARCH_SPACE_CFG = PIPE_CFG["search_spaces"].copy()
SEARCH_SPACE_CFG["clf"] = [MODELS[k] for k in MODEL_KEYS]

SEARCH_SPACES = {
    "outlier": Categorical(
        [
            None, 
            OutlierCleanerTransformer(method="clip", lower_quantile=0.01, upper_quantile=0.99),
            OutlierCleanerTransformer(method="clip", lower_quantile=0.05, upper_quantile=0.95),
           
        ],
        transform="identity",
    ),
    "scale": Categorical(
        [
            None,
            FeatureScalerTransformer(method="standard"),
            FeatureScalerTransformer(method="robust"),
        ],
        transform="identity",
    ),
    "select__method": Categorical(SEARCH_SPACE_CFG["select__method"]),
    "select__k": Categorical(SEARCH_SPACE_CFG["select__k"]),
    "reduce_pre__method": Categorical(SEARCH_SPACE_CFG["reduce_pre__method"]),
    "reduce_post__method": Categorical(SEARCH_SPACE_CFG["reduce_post__method"]),
    "clf": Categorical(
        [
            EstimatorWrapper(MODELS[k])
            for k in (
                "logreg_l2", "logreg_l1",   # Logistic Regression
                "rf_small", "rf_med", "rf_large",   # Random Forest
                "xgb_std", "xgb_hist", "xgb_deep", "xgb_shallow", "xgb_regularized",  # XGBoost
                "lda",   # Linear Discriminant Analysis
            )
        ],
        transform="identity",
    ),
}


METRICS = PIPE_CFG["metrics"]

SCORING = PIPE_CFG["scoring"]

PIPELINE_ORDER = PIPE_CFG["pipeline_order"]


def prepare_eurasian_otter() -> None:
    """Prepare splits only for the Eurasian otter dataset."""
    raw_file = RAW_DIR / "Eurasian Otter.csv"
    if not raw_file.exists():
        raise FileNotFoundError(raw_file)
    importer = DataImporter(RAW_DIR, target_cols=DEFAULT_TARGETS)
    dfs = importer.run()
    key = next(k for k in dfs if "otter" in k.lower())
    df = dfs[key]

    train_df, test_df, inf_df = create_train_test_split_otter(df)
    y_train = train_df["sex"].map({"f": 0, "m": 1})
    folds, _ = _make_folds(train_df, y_train, n_splits=NUM_FOLDS, group_col=GROUP_COL)
    train_df = train_df.assign(Fold=folds)

    out_dir = SPLITS_DIR / "eurasian_otter"
    out_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_parquet(out_dir / "train.parquet", index=False)
    test_df.to_parquet(out_dir / "test.parquet", index=False)
    inf_df.to_parquet(out_dir / "inference.parquet", index=False)

    run_summary(
        out_dir,
        RESULTS_DATA_DIR / "eurasian_otter_summary.csv",
        RESULTS_DATA_DIR / "eurasian_otter_fig",
    )


def _run_species_search(
    species: str,
    n_iter: int,
    cv: int | str,
    random_state: int,
    *,
    reuse_results: bool = PIPE_CFG.get("run_otter_search_sex", {}).get(
        "reuse_results", True
    ),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the hyperparameter search for a single species.

    Parameters
    ----------
    reuse_results:
        When ``True`` and result CSVs exist in the species search directory the
        search is skipped and the files are loaded instead.
    """
    paths = get_species_paths(species)
    search_dir = paths["search"]
    models_dir = paths["models"]
    heatmaps_dir = paths["heatmaps"]

    all_csv = search_dir / "all_results.csv"
    best_csv = search_dir / "best_models.csv"
    if reuse_results and all_csv.exists() and best_csv.exists():
        df_all = pd.read_csv(all_csv).apply(pd.to_numeric, errors="ignore")
        df_best = pd.read_csv(best_csv).apply(pd.to_numeric, errors="ignore")
        return df_all, df_best

    species_dir = SPLITS_DIR / species

    df_train = (
        pd.read_parquet(species_dir / "train.parquet")
        .query("sex in ['f','m']")
    )
    if cv == "fold":
        fold_ids = df_train["Fold"].astype(int).to_numpy()
        df_train = df_train.drop(columns=["Fold"])
        cv = PredefinedSplit(test_fold=fold_ids)
    else:
        df_train = df_train.drop(columns=["Fold"], errors="ignore")
    df_test = (
        pd.read_parquet(species_dir / "test.parquet")
        .query("sex in ['f','m']")
        .drop(columns=["Fold"], errors="ignore")
    )

    feature_cols = get_feature_cols(df_train)

    X_tr, y_tr, ids_tr = (
        df_train[feature_cols],
        df_train["sex"].map({"f": 0, "m": 1}).values,
        df_train["individual_id"].values,
    )
    X_te, y_te, ids_te = (
        df_test[feature_cols],
        df_test["sex"].map({"f": 0, "m": 1}).values,
        df_test["individual_id"].values,
    )

    steps = [
        ("transform", NumericTransformer()),
        ("outlier",   OutlierCleanerTransformer(method="clip")),
        ("scale",     FeatureScalerTransformer(method="standard")),
        ("reduce_pre", DimensionalityReducerTransformer(method=None)),       # ← new
        ("select",    FeatureSelectionTransformer(method="forward", k=1)),
        ("reduce_post", DimensionalityReducerTransformer(method=None)),      # ← new
        ("clf",       MODELS["rf_small"]),
    ]

    pipe = Pipeline(steps)

    search = BayesSearchCV(
        estimator=pipe,
        search_spaces=SEARCH_SPACES,
        n_iter=n_iter,
        scoring=SCORING,
        refit=BAYES_REFIT,
        cv=cv,
        n_jobs=-1,
        random_state=random_state,
        verbose=1,
    )

    folds = (
        search.cv
        if isinstance(search.cv, int)
        else getattr(search.cv, "n_splits", search.cv.get_n_splits())
    )
    total_fits = search.n_iter * folds
    search_dir.mkdir(parents=True, exist_ok=True)
    for m in METRICS:
        (models_dir / f"best_{m}").mkdir(parents=True, exist_ok=True)
    (models_dir / "best_mean_rank").mkdir(parents=True, exist_ok=True)

    raw_records = []
    best_records = []

    with tqdm_joblib(tqdm(desc=f"{species} BS-CV", total=total_fits, leave=False)):
        with warnings.catch_warnings(record=True) as warn_list:
            warnings.simplefilter("always", ConvergenceWarning)
            warnings.filterwarnings(
                "ignore",
                category=UserWarning,
                message="X does not have valid feature names.*",
            )
            search.fit(X_tr, y_tr)

    conv_msgs = [
        str(w.message) for w in warn_list if issubclass(w.category, ConvergenceWarning)
    ]

    cv_res = search.cv_results_
    for i, params in enumerate(cv_res["params"]):
        clean_params = {
            k: (v.estimator if isinstance(v, EstimatorWrapper) else v)
            for k, v in params.items()
        }
        record = {
            "species": species,
            "mean_test_accuracy": cv_res["mean_test_accuracy"][i],
            "mean_test_balanced_accuracy": cv_res["mean_test_balanced_accuracy"][i],
            "mean_test_neg_log_loss": cv_res["mean_test_neg_log_loss"][i],
            **clean_params,
        }

        mdl = clone(pipe).set_params(**clean_params).fit(X_tr, y_tr)
        y_tr_pred = mdl.predict(X_tr)
        y_te_pred = mdl.predict(X_te)

        # Wahrscheinlichkeiten für AUC (nur wenn vorhanden)
                # Wahrscheinlichkeiten für AUC (nur wenn vorhanden)
        try:
            y_te_proba = mdl.predict_proba(X_te)[:, 1]

            # ❌ vorher: np.vstack([X_tr, X_te])  → bricht, weil kein DF
            # ✅ neu: concat mit Pandas
            X_all = pd.concat([X_tr, X_te], axis=0)
            y_all_proba = mdl.predict_proba(X_all)[:, 1]

        except AttributeError:
            y_te_proba = None
            y_all_proba = None


        y_all_true = np.concatenate([y_tr, y_te])
        y_all_pred = np.concatenate([y_tr_pred, y_te_pred])
        ids_all = np.concatenate([ids_tr, ids_te])

        fem_tr, mal_tr, bal_tr = individual_accuracies(y_tr, y_tr_pred, ids_tr)
        fem_te, mal_te, bal_te = individual_accuracies(y_te, y_te_pred, ids_te)
        ct_tr, wr_tr, pct_tr = individual_majority_stats(y_tr, y_tr_pred, ids_tr)
        ct_te, wr_te, pct_te = individual_majority_stats(y_te, y_te_pred, ids_te)
        fem_all, mal_all, bal_all = individual_accuracies(
            y_all_true, y_all_pred, ids_all
        )
        ct_all, wr_all, pct_all = individual_majority_stats(
            y_all_true, y_all_pred, ids_all
        )

        acc_tr = accuracy_score(y_tr, y_tr_pred)
        bal_tr = balanced_accuracy_score(y_tr, y_tr_pred)
        acc_te = accuracy_score(y_te, y_te_pred)
        bal_te = balanced_accuracy_score(y_te, y_te_pred)
        acc_all = accuracy_score(y_all_true, y_all_pred)
        bal_all = balanced_accuracy_score(y_all_true, y_all_pred)

        # Neue Metriken
        f1_te = f1_score(y_te, y_te_pred)
        prec_te = precision_score(y_te, y_te_pred)
        rec_te = recall_score(y_te, y_te_pred)
        auc_te = roc_auc_score(y_te, y_te_proba) if y_te_proba is not None else float("nan")

        f1_all = f1_score(y_all_true, y_all_pred)
        prec_all = precision_score(y_all_true, y_all_pred)
        rec_all = recall_score(y_all_true, y_all_pred)
        auc_all = roc_auc_score(y_all_true, y_all_proba) if y_all_proba is not None else float("nan")

        record.update(
            {
                "female_train_acc": fem_tr,
                "male_train_acc": mal_tr,
                "balanced_train_acc": bal_tr,
                "accuracy_train": acc_tr,

                "female_test_acc": fem_te,
                "male_test_acc": mal_te,
                "balanced_test_acc": bal_te,
                "accuracy_test": acc_te,

                "f1_test": f1_te,
                "precision_test": prec_te,
                "recall_test": rec_te,
                "roc_auc_test": auc_te,

                "female_full_acc": fem_all,
                "male_full_acc": mal_all,
                "balanced_full_acc": bal_all,
                "accuracy_full": acc_all,

                "f1_full": f1_all,
                "precision_full": prec_all,
                "recall_full": rec_all,
                "roc_auc_full": auc_all,

                "maj_train_count": ct_tr,
                "maj_train_wrong": wr_tr,
                "maj_train_pct": pct_tr,
                "maj_test_count": ct_te,
                "maj_test_wrong": wr_te,
                "maj_test_pct": pct_te,
                "maj_full_count": ct_all,
                "maj_full_wrong": wr_all,
                "maj_full_pct": pct_all,
            }
        )
        pid_parts = []
        for key in PIPELINE_ORDER:
            val = clean_params.get(key)
            pid_parts.append(f"{key}={val if val is not None else 'None'}")
        record["pipeline_id"] = ";".join(pid_parts)

        for k, v in record.items():
            if pd.isna(v):
                record[k] = "None"
        raw_records.append(record)

    df_eval = pd.DataFrame([r for r in raw_records if r["species"] == species])
    for metric in METRICS:
        best_idx = df_eval[metric].idxmax()
        best_params = cv_res["params"][best_idx]
        clean_best = {
            k: (v.estimator if isinstance(v, EstimatorWrapper) else v)
            for k, v in best_params.items()
        }
        best_record = df_eval.loc[best_idx].to_dict()
        best_record["best_metric"] = metric

        best_pipe = clone(pipe).set_params(**clean_best).fit(X_tr, y_tr)
        out_path = models_dir / f"best_{metric}" / f"{species}.joblib"
        joblib.dump(best_pipe, out_path)

        print(
            f"✅ Modell für Spezies '{species}', Kriterium '{metric}' gespeichert unter:\n   {out_path}"
        )
        best_records.append(best_record)
        
    # --- NEU: Best-Overall nach Mean Rank speichern ---
    rank_metrics = [
        "mean_test_accuracy",
        "mean_test_balanced_accuracy",
        "mean_test_neg_log_loss",
        "maj_test_pct",
        "balanced_test_acc",
        "accuracy_test",
    ]
    for metric in rank_metrics:
        if "neg_log_loss" in metric:
            df_eval[metric + "_rank"] = df_eval[metric].rank(ascending=True)
        else:
            df_eval[metric + "_rank"] = df_eval[metric].rank(ascending=False)
    rank_cols = [m + "_rank" for m in rank_metrics]
    df_eval["mean_rank"] = df_eval[rank_cols].mean(axis=1)

    best_idx_rank = df_eval["mean_rank"].idxmin()
    best_params_rank = cv_res["params"][best_idx_rank]
    clean_best_rank = {
        k: (v.estimator if isinstance(v, EstimatorWrapper) else v)
        for k, v in best_params_rank.items()
    }
    best_record_rank = df_eval.loc[best_idx_rank].to_dict()
    best_record_rank["best_metric"] = "mean_rank"

    best_pipe_rank = clone(pipe).set_params(**clean_best_rank).fit(X_tr, y_tr)
    out_path_rank = models_dir / "best_mean_rank" / f"{species}.joblib"
    joblib.dump(best_pipe_rank, out_path_rank)

    print(
        f"✅ Best-Overall-Rank Modell für Spezies '{species}' gespeichert unter:\n   {out_path_rank}"
    )
    best_records.append(best_record_rank)

    all_csv = search_dir / "all_results.csv"
    best_csv = search_dir / "best_models.csv"
    df_all = pd.DataFrame(raw_records).fillna("None")
    df_best = pd.DataFrame(best_records).fillna("None")

    # Spalte is_best_overall sicherstellen
    if "is_best_overall" not in df_best.columns:
        df_best["is_best_overall"] = False

    # Best-Overall-Rank bestimmen und markieren
    if "mean_rank" in df_all.columns:
        best_idx_rank = df_all["mean_rank"].idxmin()
        df_best.loc[df_best.index == best_idx_rank, "is_best_overall"] = True

    df_all.to_csv(all_csv, index=False)
    df_best.to_csv(best_csv, index=False)

    if conv_msgs:
        counts = Counter(conv_msgs)
        for msg, cnt in counts.items():
            print(f"⚠️ {msg} (occurred {cnt} times)")

    # --- df_heat für Hyperparameter-Heatmap vorbereiten ---
    df_heat = df_all.rename(
        columns={
            "outlier": "outlier_method",
            "scale": "scaler",
            "select__method": "feature_selection_method",
            "select__k": "num_features",
           #"reduce_pre__method": "reduce_pre",
            #"reduce_post__method": "reduce_post",
            "clf": "classifier",
            "mean_test_balanced_accuracy": "cv_balanced_accuracy",
            "mean_test_accuracy": "cv_accuracy",
            "mean_test_neg_log_loss": "cv_log_loss",
        }
    )

    # Klassifikatornamen sauber extrahieren (statt Wrapper)
    if "classifier" in df_heat.columns:
        df_heat["classifier"] = df_heat["classifier"].apply(
            lambda x: getattr(x, "estimator", x).__class__.__name__ if x is not None else "None"
        )

    # Outlier Methoden verständlicher machen
    if "outlier_method" in df_heat.columns:
        df_heat["outlier_method"] = df_heat["outlier_method"].apply(
            lambda x: (
                f"clip_{x.lower_quantile:.2f}_{x.upper_quantile:.2f}"
                if hasattr(x, "method") and x.method == "clip"
                else ("None" if x is None else getattr(x, "method", str(x)))
            )
        )

    plot_hyperparam_heatmap(df_heat, heatmaps_dir / "hyperparam_search")

    # Create prediction CSVs for downstream pipelines
    predict_all(
        species,
        models_dir=models_dir,
        reuse_csv=False,
        prefer_generic=True,
        include_inference=False,
        metric_key=SEX_PREDICT_METRIC,
    )

    return df_all, df_best




def run_otter_search_sex(
    n_iter: int = PIPE_CFG["run_otter_search_sex"]["n_iter"],
    cv: int | str = PIPE_CFG["run_otter_search_sex"]["cv"],
    random_state: int = PIPE_CFG["run_otter_search_sex"]["random_state"],
    *,
    reuse_results: bool = PIPE_CFG.get("run_otter_search_sex", {}).get(
        "reuse_results", True
    ),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run BayesSearchCV for the Eurasian otter dataset."""

    return _run_species_search(
        "eurasian_otter",
        n_iter,
        cv,
        random_state,
        reuse_results=reuse_results,
    )


def run_species_search(
    species_filter: list[str] | None = None,
    n_iter: int = PIPE_CFG["run_otter_search_sex"]["n_iter"],
    cv: int | str = PIPE_CFG["run_otter_search_sex"]["cv"],
    random_state: int = PIPE_CFG["run_otter_search_sex"]["random_state"],
    *,
    reuse_results: bool = PIPE_CFG.get("run_otter_search_sex", {}).get(
        "reuse_results", True
    ),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the search for all species in ``SPLITS_DIR``."""
    all_dfs: list[pd.DataFrame] = []
    best_dfs: list[pd.DataFrame] = []
    for species_dir in sorted(SPLITS_DIR.iterdir()):
        if not species_dir.is_dir():
            continue
        species = species_dir.name
        if species_filter and species not in species_filter:
            continue
        df_all, df_best = _run_species_search(
            species,
            n_iter,
            cv,
            random_state,
            reuse_results=reuse_results,
        )
        all_dfs.append(df_all)
        best_dfs.append(df_best)

    df_all_comb = pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()
    df_best_comb = pd.concat(best_dfs, ignore_index=True) if best_dfs else pd.DataFrame()
    return df_all_comb, df_best_comb

def run_other_species_search(
    n_iter: int = PIPE_CFG["run_otter_search_sex"]["n_iter"],
    cv: int | str = PIPE_CFG["run_otter_search_sex"]["cv"],
    random_state: int = PIPE_CFG["run_otter_search_sex"]["random_state"],
) -> None:
    """Run the search for all species except the Eurasian otter.

    Parameters
    ----------
    n_iter : int, optional
        Number of iterations for :class:`skopt.BayesSearchCV`.
    cv : int or str, optional
        Cross-validation strategy forwarded to :func:`run_species_search`.
    random_state : int, optional
        Random seed controlling the search.
    """

    run_species_search(
        species_filter=[s.name for s in SPLITS_DIR.iterdir() if s.is_dir() and s.name != "eurasian_otter"],
        n_iter=n_iter,
        cv=cv,
        random_state=random_state,
    )

