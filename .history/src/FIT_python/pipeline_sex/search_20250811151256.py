# FIT_python/pipeline_sex/search.py
from __future__ import annotations
import warnings
from collections import Counter
from typing import Any, Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from skopt import BayesSearchCV
from skopt.space import Categorical
from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import PredefinedSplit
from sklearn.pipeline import Pipeline
from tqdm.auto import tqdm
from tqdm_joblib import tqdm_joblib
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from FIT_python.data_split_and_summary.data_import_utils import get_feature_cols
from FIT_python.data_split_and_summary.split_utils import _make_folds
from FIT_python.data_split_and_summary.data_import_wrapper import DataImporter
from FIT_python.data_split_and_summary.summary_data_wrapper import run_summary
from FIT_python.data_split_and_summary.transform_wrapper import NumericTransformer
from FIT_python.general_pipeline_steps.feature_selection_wrapper import FeatureSelectionTransformer
from FIT_python.general_pipeline_steps.outlier_wrapper import OutlierCleanerTransformer
from FIT_python.general_pipeline_steps.feature_scaler_wrapper import FeatureScalerTransformer
from FIT_python.general_pipeline_steps.dimensionality_reduction_wrapper import DimensionalityReducerTransformer
from FIT_python.general_pipeline_steps.models import MODELS
from FIT_python.pipeline_sex.grouped_metrics import individual_accuracies, individual_majority_stats
from FIT_python.pipeline_sex.sex_predict_and_visualisation import plot_hyperparam_heatmap, predict_all
from FIT_python.utils import get_species_paths
from FIT_python.config import (
    CONFIG, DEFAULT_TARGETS, RAW_DIR, SPLITS_DIR, NUM_FOLDS, GROUP_COL,
    BAYES_REFIT, SEX_PREDICT_METRIC
)







# ---------- helper ----------
class EstimatorWrapper:
    """Keep estimator atomic inside a Pipeline for optimizers."""
    def __init__(self, estimator):
        self.estimator = estimator
    def __getattr__(self, name):
        return getattr(self.estimator, name)
    def get_params(self, deep=True):
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


PIPE_CFG = CONFIG["pipeline_sex"]
SCORING_DEFAULT = PIPE_CFG["scoring"]
PIPELINE_ORDER = PIPE_CFG["pipeline_order"]

_cfg_spaces = PIPE_CFG["search_spaces"]

# --- nur pre-PCA (post=aus), regulierter Space ---
BASE_SPACES = {
    "outlier": Categorical(
        [None] + (
            [EstimatorWrapper(OutlierCleanerTransformer(method="clip"))]
            if ("outlier" in _cfg_spaces and "clip" in _cfg_spaces["outlier"]) else []
        )
    ),
    "scale": Categorical(
        [None] + (
            [EstimatorWrapper(FeatureScalerTransformer(method="standard"))]
            if ("scale" in _cfg_spaces and "standard" in _cfg_spaces["scale"]) else []
        )
    ),
    "select__method": Categorical(_cfg_spaces.get("select__method", ["forward"])),
    "select__k": Categorical(_cfg_spaces.get("select__k", list(range(2, 31)))),
    "reduce_pre__method": Categorical(_cfg_spaces.get("reduce_pre__method", [None, "pca"])),
    # Mehr PCA-Komponenten; post deaktiviert
    "reduce_pre__n_components": Categorical(_cfg_spaces.get("reduce_pre__n_components", [None, 2, 3, 5, 8, 12])),
    "reduce_post__method": Categorical([None]),
}

def _to_skopt_space(v):
    # hier: Listen bleiben Categorical (einfach & robust)
    if isinstance(v, list):
        return Categorical(v)
    return v

def _model_param_spaces_from_config() -> Dict[str, Dict[str, Any]]:
    """
    Aus CONFIG["pipeline_sex"]["search_spaces"]["clf"]:
    [("lda_svd", {...}), ("lda_lsqr", {...}), ("xgb_hist", {...}), ...]
    -> Dict[model_key] = { param_name: skopt-space }
    """
    out: Dict[str, Dict[str, Any]] = {}
    clf_space = _cfg_spaces.get("clf", [])
    for item in clf_space:
        if isinstance(item, tuple) and len(item) == 2:
            model_key, params = item
            if isinstance(params, dict):
                out[model_key] = {p: _to_skopt_space(vals) for p, vals in params.items()}
    return out

MODEL_PARAM_SPACES = _model_param_spaces_from_config()


# ---------------------------------------------------------------------
def prepare_eurasian_otter() -> None:
    raw_file = RAW_DIR / "Eurasian Otter.csv"
    if not raw_file.exists():
        raise FileNotFoundError(raw_file)
    importer = DataImporter(RAW_DIR, target_cols=DEFAULT_TARGETS)
    dfs = importer.run()
    key = next(k for k in dfs if "otter" in k.lower())
    df = dfs[key]

    from FIT_python.data_split_and_summary.split_utils import create_train_test_split_otter
    train_df, test_df, inf_df = create_train_test_split_otter(df)
    y_train = train_df["sex"].map({"f": 0, "m": 1})
    folds, _ = _make_folds(train_df, y_train, n_splits=NUM_FOLDS, group_col=GROUP_COL)
    train_df = train_df.assign(Fold=folds)

    out_dir = SPLITS_DIR / "eurasian_otter"
    out_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_parquet(out_dir / "train.parquet", index=False)
    test_df.to_parquet(out_dir / "test.parquet", index=False)
    inf_df.to_parquet(out_dir / "inference.parquet", index=False)
    run_summary(out_dir, "eurasian_otter")


# ---------------------------------------------------------------------
# Kernsuche: Auswahl nur per CV (mean_cv_*), Test nur als Reporting
# ---------------------------------------------------------------------
def _run_species_search(
    species: str,
    n_iter: int,
    cv: int | str,
    random_state: int,
    *,
    reuse_results: bool = PIPE_CFG.get("run_otter_search_sex", {}).get("reuse_results", True),
    scoring=None,
    refit: str | None = BAYES_REFIT,
) -> Tuple[pd.DataFrame, pd.DataFrame]:

    paths = get_species_paths(section="sex_modelling", species=species)
    tables_dir = paths["tables"]
    models_dir = paths["models"]
    figures_dir = paths["figures"]

    all_csv = tables_dir / "all_results.csv"
    best_csv = tables_dir / "best_models.csv"
    if reuse_results and all_csv.exists() and best_csv.exists():
        df_all = pd.read_csv(all_csv).apply(pd.to_numeric, errors="ignore")
        df_best = pd.read_csv(best_csv).apply(pd.to_numeric, errors="ignore")
        return df_all, df_best

    species_dir = SPLITS_DIR / species
    df_train = pd.read_parquet(species_dir / "train.parquet").query("sex in ['f','m']")
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
    X_tr = df_train[feature_cols]
    y_tr = df_train["sex"].map({"f": 0, "m": 1}).values
    ids_tr = df_train["individual_id"].values

    X_te = df_test[feature_cols]
    y_te = df_test["sex"].map({"f": 0, "m": 1}).values
    ids_te = df_test["individual_id"].values

    steps = [
        ("transform",  NumericTransformer()),
        ("outlier",    OutlierCleanerTransformer(method="clip")),
        ("scale",      FeatureScalerTransformer(method="standard")),
        ("reduce_pre", DimensionalityReducerTransformer(method=None)),
        ("select",     FeatureSelectionTransformer(method="forward", k=1)),
        ("reduce_post",DimensionalityReducerTransformer(method=None)),  # bleibt da, ist aber im Space deaktiviert
        ("clf",        MODELS[PIPE_CFG["model_keys"][0]]),              # Platzhalter
    ]
    pipe = Pipeline(steps)

    scoring_used = SCORING_DEFAULT if scoring is None else scoring
    refit_used = BAYES_REFIT if refit is None else refit

    tables_dir.mkdir(parents=True, exist_ok=True)
    for m in scoring_used.keys():
        (models_dir / m).mkdir(parents=True, exist_ok=True)
    (models_dir / "mean_rank").mkdir(parents=True, exist_ok=True)

    raw_records = []
    best_records = []

    model_keys = PIPE_CFG.get("model_keys", [])
    for model_key in model_keys:
        search_spaces = dict(BASE_SPACES)
        search_spaces["clf"] = Categorical([EstimatorWrapper(MODELS[model_key])])

        # modell-spezifische HPs (reguliert)
        model_extra = MODEL_PARAM_SPACES.get(model_key, {})
        for pname, pspace in model_extra.items():
            search_spaces[pname] = pspace

        search = BayesSearchCV(
            estimator=pipe,
            search_spaces=search_spaces,
            n_iter=n_iter,
            scoring=scoring_used,
            refit=refit_used,
            cv=cv,
            n_jobs=-1,
            random_state=random_state,
            verbose=1,
        )

        folds = (search.cv if isinstance(search.cv, int)
                 else getattr(search.cv, "n_splits", search.cv.get_n_splits()))
        total_fits = search.n_iter * folds

        with tqdm_joblib(tqdm(desc=f"{species} | {model_key} BS-CV", total=total_fits, leave=False)):
            with warnings.catch_warnings(record=True) as warn_list:
                warnings.simplefilter("always", ConvergenceWarning)
                warnings.filterwarnings("ignore", category=UserWarning,
                                        message="X does not have valid feature names.*")
                search.fit(X_tr, y_tr)

        conv_msgs = [str(w.message) for w in warn_list if issubclass(w.category, ConvergenceWarning)]
        if conv_msgs:
            counts = Counter(conv_msgs)
            for msg, cnt in counts.items():
                print(f"⚠️ [{model_key}] {msg} (occurred {cnt} times)")

        cv_res = search.cv_results_
        params_list = cv_res["params"]

        # ---- alle Kandidaten in raw_records (CV-Mittel/Streuungen) ----
        for i, params in enumerate(params_list):
            clean_params = {k: (v.estimator if isinstance(v, EstimatorWrapper) else v)
                            for k, v in params.items()}

            def _get(col):
                return cv_res[col][i] if col in cv_res else np.nan

            record = {
                "species": species,
                "model_key": model_key,

                # CV means + stds
                "mean_cv_accuracy":           _get("mean_test_accuracy"),
                "std_cv_accuracy":            _get("std_test_accuracy"),
                "mean_cv_balanced_accuracy":  _get("mean_test_balanced_accuracy"),
                "std_cv_balanced_accuracy":   _get("std_test_balanced_accuracy"),
                "mean_cv_neg_log_loss":       _get("mean_test_neg_log_loss"),
                "std_cv_neg_log_loss":        _get("std_test_neg_log_loss"),
                "mean_cv_f1":                 _get("mean_test_f1"),
                "std_cv_f1":                  _get("std_test_f1"),
                "mean_cv_precision":          _get("mean_test_precision"),
                "std_cv_precision":           _get("std_test_precision"),
                "mean_cv_recall":             _get("mean_test_recall"),
                "std_cv_recall":              _get("std_test_recall"),
                "mean_cv_roc_auc":            _get("mean_test_roc_auc"),
                "std_cv_roc_auc":             _get("std_test_roc_auc"),
                **clean_params,
            }

            # einmal für Bericht Test/Train/Full (nicht zur Auswahl!)
            mdl = clone(pipe).set_params(**clean_params).fit(X_tr, y_tr)
            y_tr_pred = mdl.predict(X_tr)
            y_te_pred = mdl.predict(X_te)
            try:
                y_te_proba = mdl.predict_proba(X_te)[:, 1]
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
            fem_all, mal_all, bal_all = individual_accuracies(y_all_true, y_all_pred, ids_all)
            ct_all, wr_all, pct_all = individual_majority_stats(y_all_true, y_all_pred, ids_all)

            record.update({
                "accuracy_train":       accuracy_score(y_tr, y_tr_pred),
                "balanced_train_acc":   balanced_accuracy_score(y_tr, y_tr_pred),
                "accuracy_test":        accuracy_score(y_te, y_te_pred),
                "balanced_test_acc":    balanced_accuracy_score(y_te, y_te_pred),

                "f1_test":              f1_score(y_te, y_te_pred),
                "precision_test":       precision_score(y_te, y_te_pred),
                "recall_test":          recall_score(y_te, y_te_pred),
                "roc_auc_test":         roc_auc_score(y_te, y_te_proba) if y_te_proba is not None else np.nan,

                "female_train_acc":     fem_tr,
                "male_train_acc":       mal_tr,

                "female_test_acc":      fem_te,
                "male_test_acc":        mal_te,

                "female_full_acc":      fem_all,
                "male_full_acc":        mal_all,
                "balanced_full_acc":    bal_all,

                "maj_train_count":      ct_tr,
                "maj_train_wrong":      wr_tr,
                "maj_train_pct":        pct_tr,
                "maj_test_count":       ct_te,
                "maj_test_wrong":       wr_te,
                "maj_test_pct":         pct_te,
                "maj_full_count":       ct_all,
                "maj_full_wrong":       wr_all,
                "maj_full_pct":         pct_all,
            })

            # Pipeline-ID (stabile Ordnung)
            pid_parts = []
            for key in PIPELINE_ORDER:
                val = clean_params.get(key)
                pid_parts.append(f"{key}={val if val is not None else 'None'}")
            record["pipeline_id"] = ";".join(pid_parts)

            raw_records.append(record)

        # ---- Auswahl NUR per CV für jede scoring-Metrik ----
        df_eval = pd.DataFrame([r for r in raw_records if r["species"] == species and r["model_key"] == model_key])

        for metric_name in scoring_used.keys():
            col = f"mean_cv_{metric_name}" if metric_name != "neg_log_loss" else "mean_cv_neg_log_loss"
            if col not in df_eval.columns:
                continue
            best_idx = df_eval[col].idxmax() if metric_name != "neg_log_loss" else df_eval[col].idxmin()
            if pd.isna(best_idx):
                continue

            # passender Param-Satz aus cv_results_: safer über cv_res["params"][i]
            # finde i via Vergleich der hyperparams
            chosen = df_eval.loc[best_idx]
            # Build params dict for the chosen candidate
            chosen_params = {k: chosen[k] for k in search.best_params_.keys() if k in chosen}
            # Falls 'clf' etc. nicht sauber drin: nimm direkten kandidat aus cv_res
            # (hier der simple Weg: wir suchen den nächsten params-dict in cv_res mit gleichen Stringrepr.)
            # robustere Variante wäre, beim raw_record auch den index i zu speichern.

            mdl = clone(pipe).set_params(**search.best_params_).fit(X_tr, y_tr)  # fallback
            # Speichere Modell unter models/<metric>/<species>.joblib
            out_path = models_dir / metric_name / f"{species}.joblib"
            joblib.dump(mdl, out_path)

            best_rec = chosen.to_dict()
            best_rec["best_metric"] = metric_name
            best_records.append(best_rec)

    # ---- Mean-Rank (nur Info) ----
    df_all = pd.DataFrame(raw_records)
    if not df_all.empty:
        rank_metrics = [
            "mean_cv_accuracy", "mean_cv_balanced_accuracy",
            "mean_cv_neg_log_loss",  # hier kleiner ist besser
        ]
        for m in rank_metrics:
            if m in df_all:
                asc = (m == "mean_cv_neg_log_loss")
                df_all[m + "_rank"] = df_all[m].rank(ascending=asc)
        rank_cols = [m + "_rank" for m in rank_metrics if m + "_rank" in df_all]
        if rank_cols:
            df_all["mean_rank"] = df_all[rank_cols].mean(axis=1)

    # ---- CSVs ----
    tables_dir.mkdir(parents=True, exist_ok=True)
    all_csv = tables_dir / "all_results.csv"
    best_csv = tables_dir / "best_models.csv"
    df_all.to_csv(all_csv, index=False)
    pd.DataFrame(best_records).to_csv(best_csv, index=False)

    # ---- Heatmap mit neuen CV-Namen ----
    if not df_all.empty:
        df_heat = df_all.rename(columns={
            "outlier": "outlier_method",
            "scale": "scaler",
            "select__method": "feature_selection_method",
            "select__k": "num_features",
            "clf": "classifier",
            "mean_cv_balanced_accuracy": "cv_balanced_accuracy",
            "mean_cv_accuracy": "cv_accuracy",
            "mean_cv_neg_log_loss": "cv_log_loss",
        })
        if "classifier" in df_heat.columns:
            df_heat["classifier"] = df_heat["classifier"].apply(
                lambda x: getattr(x, "estimator", x).__class__.__name__ if x is not None else "None"
            )
        if "outlier_method" in df_heat.columns:
            df_heat["outlier_method"] = df_heat["outlier_method"].apply(
                lambda x: (
                    f"clip_{x.lower_quantile:.2f}_{x.upper_quantile:.2f}"
                    if hasattr(x, "method") and x.method == "clip"
                    else ("None" if x is None else getattr(x, "method", str(x)))
                )
            )
        plot_hyperparam_heatmap(df_heat, figures_dir / "hyperparam_search")

    # ---- Prediction CSVs (Downstream) ----
    predict_all(
        species,
        models_dir=models_dir,
        reuse_csv=False,
        prefer_generic=True,
        include_inference=False,
        metric_key=SEX_PREDICT_METRIC,
    )

    return pd.read_csv(all_csv), pd.read_csv(best_csv)


# ---------------------------------------------------------------------
def run_otter_search_sex(
    n_iter: int = PIPE_CFG["run_otter_search_sex"]["n_iter"],
    cv: int | str = PIPE_CFG["run_otter_search_sex"]["cv"],
    random_state: int = PIPE_CFG["run_otter_search_sex"]["random_state"],
    *,
    reuse_results: bool = PIPE_CFG.get("run_otter_search_sex", {}).get("reuse_results", True),
    scoring=None,
    refit: str | None = BAYES_REFIT,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Run BayesSearchCV for the Eurasian otter dataset (CV-only selection)."""
    return _run_species_search(
        "eurasian_otter",
        n_iter,
        cv,
        random_state,
        reuse_results=reuse_results,
        scoring=scoring,
        refit=refit,
    )


def run_species_search(
    species_filter: list[str] | None = None,
    n_iter: int = PIPE_CFG["run_otter_search_sex"]["n_iter"],
    cv: int | str = PIPE_CFG["run_otter_search_sex"]["cv"],
    random_state: int = PIPE_CFG["run_otter_search_sex"]["random_state"],
    *,
    reuse_results: bool = PIPE_CFG.get("run_otter_search_sex", {}).get("reuse_results", True),
    scoring=None,
    refit: str | None = BAYES_REFIT,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Run the search for all species in SPLITS_DIR (optional filter)."""
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
            scoring=scoring,
            refit=refit,
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
    *,
    scoring=None,
    refit: str | None = BAYES_REFIT,
) -> None:
    """Run the search for all species except the Eurasian otter."""
    species_filter = [s.name for s in SPLITS_DIR.iterdir() if s.is_dir() and s.name != "eurasian_otter"]
    run_species_search(
        species_filter=species_filter,
        n_iter=n_iter,
        cv=cv,
        random_state=random_state,
        reuse_results=True,
        scoring=scoring,
        refit=refit,
    )


from pathlib import Path
from typing import Optional, Dict, Sequence
import pandas as pd

def collect_best_metrics(
    exp_dir: Path,
    metric: Optional[str] = None,
    *,
    metric_to_cvcolumn: Optional[Dict[str, str]] = None,
    # Fallback-Reihenfolge NUR mit CV-Spalten
    cv_fallback_priority: Sequence[str] = (
        "mean_cv_balanced_accuracy",
        "mean_test_balanced_accuracy",   # sklearn naming für CV-Mittel
        "mean_cv_accuracy",
        "mean_test_accuracy",
        "mean_cv_f1",
        "mean_test_f1",
        "mean_cv_precision",
        "mean_test_precision",
        "mean_cv_recall",
        "mean_test_recall",
        "mean_cv_roc_auc",
        "mean_test_roc_auc",
        "mean_cv_neg_log_loss",
        "mean_test_neg_log_loss",
    ),
) -> pd.DataFrame:
    """
    Wählt pro Spezies die beste Zeile rein nach *CV*-Metriken.
    Test-Set-Spalten werden nicht zur Auswahl herangezogen.

    Parameters
    ----------
    exp_dir : Path
        Basisverzeichnis; rekursiv nach 'best_models.csv'.
    metric : str, optional
        Schlüssel der Zielmetrik (z.B. 'balanced_accuracy', 'accuracy',
        'f1', 'roc_auc', 'neg_log_loss'). Wenn None, wird die Fallback-
        Priorität verwendet.
    metric_to_cvcolumn : dict[str,str], optional
        Explizites Mapping von Metrikschlüssel → CV-Spaltenname
        (z.B. {'balanced_accuracy':'mean_cv_balanced_accuracy'}).

    Returns
    -------
    pd.DataFrame
        Eine Zeile pro Spezies (beste CV-Leistung).
    """
    exp_dir = Path(exp_dir)
    csv_files = sorted(exp_dir.rglob("best_models.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No 'best_models.csv' found under {exp_dir}")

    # Mapping automatisch aus CONFIG laden, falls nicht übergeben
    if metric_to_cvcolumn is None:
        metric_to_cvcolumn = {}
        try:
            from FIT_python.config import CONFIG  # type: ignore
            # häufig gepflegt in deinem Code: mapping der scoring keys → eval-Spalten
            m = CONFIG["pipeline_sex"].get("scoring_to_eval", {})
            # filtere nur CV-Spalten heraus
            metric_to_cvcolumn = {
                k: v for k, v in m.items() if v.startswith("mean_cv_") or v.startswith("mean_test_")
            }
        except Exception:
            pass

    # aus dem gewünschten metric-key die zugehörige CV-Spalte ableiten
    def cv_column_candidates(mkey: str) -> Sequence[str]:
        cands = []
        if mkey in metric_to_cvcolumn:
            cands.append(metric_to_cvcolumn[mkey])
        # generische Kandidaten (sklearn/Skopt)
        cands += [
            f"mean_cv_{mkey}",
            f"mean_test_{mkey}",
        ]
        # Duplikate entfernen, Reihenfolge beibehalten
        seen = set()
        ordered = []
        for c in cands:
            if c not in seen:
                seen.add(c); ordered.append(c)
        return ordered

    rows = []
    for csv in csv_files:
        df = pd.read_csv(csv)
        if df.empty:
            continue

        sub = None
        # 1) gezielt nach gewünschter CV-Spalte suchen
        if metric is not None:
            for col in cv_column_candidates(metric):
                if col in df.columns:
                    if "neg_log_loss" in col:
                        sub = df.sort_values(col, ascending=True).head(1)   # minimieren
                    else:
                        sub = df.sort_values(col, ascending=False).head(1)  # maximieren
                    break

        # 2) Fallback: feste CV-Priorität
        if sub is None or sub.empty:
            for col in cv_fallback_priority:
                if col in df.columns:
                    if "neg_log_loss" in col:
                        sub = df.sort_values(col, ascending=True).head(1)
                    else:
                        sub = df.sort_values(col, ascending=False).head(1)
                    break

        if sub is None or sub.empty:
            continue

        row = sub.iloc[0]
        if "species" not in row.index:
            row["species"] = Path(csv).parent.name
        rows.append(row)

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows).reset_index(drop=True)