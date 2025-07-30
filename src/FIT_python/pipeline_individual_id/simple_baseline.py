from __future__ import annotations


"""Utility helpers for a simple ID baseline."""

from pathlib import Path
from typing import Iterable, List, Dict, Any
import warnings
from tqdm.auto import tqdm

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from FIT_python.caption_utils import save_caption
from FIT_python.Visualisations.plot_style import apply_style
from .population_estimation import concordance_correlation_coefficient
from FIT_python.config import SPLITS_DIR, PATHS
from FIT_python.soft_config import SOFT_CONFIG
from FIT_python.data_split_and_summary.data_import_utils import get_feature_cols
from . import sequential_holdout
from FIT_python.pipeline_sex.sex_predict_and_visualisation import predict_all


def _pairs_to_array(
    trail_a: Iterable[str], trail_b: Iterable[str], distances: Iterable[float]
) -> tuple[np.ndarray, list[str]]:
    """Return distance array and trail order via index mapping.

    Using the vectorised approach is about an order of magnitude faster
    than iterating over ``DataFrame`` rows.
    """

    trails = pd.Index(sorted(set(trail_a) | set(trail_b)), dtype=str)
    index = pd.Series(np.arange(len(trails)), index=trails)
    ia = index.loc[pd.Index(trail_a)].to_numpy()
    ib = index.loc[pd.Index(trail_b)].to_numpy()
    vals = pd.to_numeric(pd.Series(distances), errors="coerce").to_numpy()

    arr = np.full((len(trails), len(trails)), np.nan)
    mask = ~np.isnan(vals)
    arr[ia[mask], ib[mask]] = vals[mask]
    arr[ib[mask], ia[mask]] = vals[mask]
    np.fill_diagonal(arr, 0.0)
    return arr, trails.to_list()


def collect_id_metrics(exp_dir: Path) -> pd.DataFrame:
    """Return averaged metrics across baseline holdout splits.

    Parameters
    ----------
    exp_dir:
        Root directory containing one subdirectory per species with a
        ``summary.csv`` produced by the baseline helper.

    Returns
    -------
    pandas.DataFrame
        Table with one row per species containing the mean BCR, mean ERD
        and the concordance correlation coefficient (CCC).
    """
    exp_dir = Path(exp_dir)
    csv_files = sorted(exp_dir.glob("*/summary.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No 'summary.csv' found under {exp_dir}")

    records: List[Dict[str, Any]] = []
    for csv in csv_files:
        df = pd.read_csv(csv)
        if df.empty:
            continue
        mean_bcr = df["bcr"].mean() if "bcr" in df.columns else float("nan")
        mean_erd = df["erd"].mean() if "erd" in df.columns else float("nan")
        if "ccc" in df.columns and not df["ccc"].isna().all():
            ccc = float(df["ccc"].iloc[0])
        elif {"pred_count", "true_count"}.issubset(df.columns):
            ccc = concordance_correlation_coefficient(
                df["pred_count"], df["true_count"]
            )
        else:
            ccc = float("nan")
        records.append(
            {"species": csv.parent.name, "bcr": mean_bcr, "erd": mean_erd, "ccc": ccc}
        )

    result = pd.DataFrame(records)
    result.to_csv(exp_dir / "raw_results.csv", index=False)
    return result


def plot_bcr_comparison(df: pd.DataFrame, fig_dir: Path) -> Path:
    """Plot a bar chart comparing BCR across species."""
    if "species" not in df.columns or "bcr" not in df.columns:
        raise KeyError("DataFrame must contain 'species' and 'bcr'")

    apply_style()
    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    order = df.sort_values("bcr", ascending=False)["species"]

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(
        data=df,
        x="species",
        y="bcr",
        order=order,
        ax=ax,
        color="#4C72B0",
        edgecolor="black",
    )
    ax.set_xlabel("Species")
    ax.set_ylabel("BCR")
    plt.xticks(rotation=45, ha="right")
    fig.tight_layout()

    out = fig_dir / "bcr_comparison.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    save_caption(out, "Baseline BCR per species")
    return out


def run_simple_baseline_otter(
    exp_dir: Path,
    k_range: Iterable[int] = range(12, 21),
    iterations: int = 10,
    use_sex_predictions: bool = False,
    *,
    reuse_results: bool = True,
) -> int:
    """Run sequential holdouts for the otter dataset.

    The helper evaluates different feature counts using
    :func:`sequential_holdout.run` and returns the ``k`` with the best mean BCR.

    Parameters
    ----------
    exp_dir : Path
        Directory used to store intermediate and summary results.
    k_range : Iterable[int], optional
        Range of feature counts to evaluate.  Defaults to ``range(12, 21)``.
    iterations : int, optional
        Number of sequential holdout iterations per ``k``.  Defaults to ``10``.
    use_sex_predictions : bool, optional
        Append sex-model probabilities via :func:`load_sex_predictions` when
        ``True``.
    reuse_results : bool, optional
        When ``True`` already existing summaries are loaded instead of
        recomputing the baseline.

    Returns
    -------
    int
        The ``k`` value that achieved the best balanced correct recognition
        rate.
    """

    exp_dir = Path(exp_dir)
    out_dir = exp_dir / "otter"
    out_dir.mkdir(parents=True, exist_ok=True)

    species_dir = SPLITS_DIR / "eurasian_otter"
    df = _load_splits(species_dir, include_test=False)
    feature_cols = get_feature_cols(df)

    sex_preds = load_sex_predictions(species_dir.name) if use_sex_predictions else None

    best_k: int | None = None
    best_bcr = float("-inf")

    best_summary: pd.DataFrame | None = None

    # check if all results already exist
    if reuse_results:
        loaded = {}
        summaries: dict[int, pd.DataFrame] = {}
        for k in k_range:
            k_dir = out_dir / f"k{k}"
            sum_fp = k_dir / "summary.csv"
            agg_fp = out_dir / f"summary_k{k}.csv"
            if sum_fp.exists():
                df_sum = pd.read_csv(sum_fp)
                summaries[k] = df_sum
            elif agg_fp.exists():
                df_sum = pd.read_csv(agg_fp)
            else:
                loaded = None
                break
            mean_bcr = (
                float(df_sum["bcr"].mean()) if "bcr" in df_sum.columns else float("nan")
            )
            loaded[k] = mean_bcr
        if loaded is not None and loaded:
            best_k = max(loaded, key=loaded.get)
            best_bcr = loaded[best_k]
            best_summary = summaries.get(best_k)
            if use_sex_predictions and best_summary is not None:
                best_summary.to_csv(out_dir / "summary_with_sex.csv", index=False)
            return best_k

    for k in k_range:
        k_dir = out_dir / f"k{k}"
        sum_fp = k_dir / "summary.csv"
        agg_fp = out_dir / f"summary_k{k}.csv"

        if reuse_results and sum_fp.exists():
            summary = pd.read_csv(sum_fp)
        elif reuse_results and agg_fp.exists():
            summary = pd.read_csv(agg_fp)
        else:
            summary = sequential_holdout.run(
                df,
                feature_cols,
                sex_predictions=sex_preds,
                iterations=iterations,
                out_dir=k_dir,
                k_features=k,
                trail_col="Trail",
                subsample=False,
            )

        agg = summary[["bcr", "pred_count", "true_count", "erd", "ccc"]].mean()
        pd.DataFrame([agg]).to_csv(agg_fp, index=False)

        mean_bcr = float(agg.get("bcr", float("nan")))
        if mean_bcr > best_bcr:
            best_bcr = mean_bcr
            best_k = k
            if "split" in summary.columns:
                best_summary = summary

    if best_k is None:
        raise RuntimeError("No valid results computed for otter baseline")

    if use_sex_predictions and best_summary is not None:
        best_summary.to_csv(out_dir / "summary_with_sex.csv", index=False)

    return best_k


def run_baseline_all_species(
    exp_dir: Path,
    best_k: int,
    cutoff: Dict[str, Any],
    *,
    reuse_summary: bool = True,
    n_jobs: int = -1,
) -> None:
    """Evaluate the baseline for every species using sequential holdouts.

    For each species in :data:`SPLITS_DIR` this helper invokes
    :func:`sequential_holdout.run` once and writes ``summary.csv`` to
    ``exp_dir/<species>``.

    Parameters
    ----------
    exp_dir : Path
        Root directory for all results. One subdirectory per species will be
        created underneath this path.
    best_k : int
        Feature count used for the otter dataset and as the default for all
        other species.
    cutoff : dict[str, Any]
        Mapping of species name to a parameter dictionary with optional keys
        ``"k"`` (int) and ``"ward"`` (float) overriding ``best_k`` and the Ward
        clustering cut-off.
    reuse_summary : bool, optional
        When ``True`` and ``exp_dir/<species>/summary.csv`` exists, the
        computation for that species is skipped.
    n_jobs : int, optional
        Parallel jobs forwarded to :func:`sequential_holdout.run`.  ``-1`` uses
        all available CPU cores.
    """

    exp_dir = Path(exp_dir)
    exp_dir.mkdir(parents=True, exist_ok=True)

    if selection_method is None:
        selection_method = SOFT_CONFIG["pipeline_individual_id"][
            "pairwise_defaults"
        ]["selection_method"]
    if reducers is None:
        reducers = SOFT_CONFIG["pipeline_individual_id"]["pairwise_defaults"][
            "reducers"
        ]
    if n_components is None:
        n_components = SOFT_CONFIG["pipeline_individual_id"]["pairwise_defaults"][
            "n_components"
        ]
    if scaler_methods is None:
        scaler_methods = SOFT_CONFIG["pipeline_individual_id"]["pairwise_defaults"][
            "scaler_methods"
        ]

    for species_dir in tqdm(sorted(SPLITS_DIR.iterdir()), desc="Species"):
        if not species_dir.is_dir():
            continue

        out_dir = exp_dir / species_dir.name
        summary_fp = out_dir / "summary.csv"
        if reuse_summary and summary_fp.exists():
            continue

        # only use the training split to derive sequential holdouts
        try:
            df = _load_splits(species_dir, include_test=False)
        except FileNotFoundError:
            warnings.warn(
                f"Split directory for {species_dir.name!r} not found \u2013 skipping.",
                UserWarning,
            )
            continue
        feature_cols = get_feature_cols(df)

        if species_dir.name == "eurasian_otter":
            k = best_k
            ward = None
        else:
            spec_cfg = cutoff.get(species_dir.name, {})
            k = spec_cfg.get("k", best_k)
            ward = spec_cfg.get("ward")

        sequential_holdout.run(
            df,
            feature_cols,
            iterations=1,
            out_dir=out_dir,
            k_features=k,
            trail_col="Trail",
            subsample=False,
            cutoff=ward,
            reuse_summary=reuse_summary,
            n_jobs=n_jobs,
        )


def run_simple_baseline_all_species(
    exp_dir: Path,
    best_k: int,
    cutoff: Dict[str, Any],
    *,
    subsample: bool = False,
    reuse_summary: bool = True,
    n_jobs: int = -1,
    use_sex_predictions: bool = False,
    use_sexmodel_prediction: bool = False,
    models_dir: Path | None = None,
    sexmodel_path: str | Path | None = None,
    selection_method: str | None = None,
    reducers: Iterable[str] | str | None = None,
    n_components: int | Iterable[int] | None = None,
    scaler_methods: Iterable[str] | str | None = None,
) -> None:
    """Evaluate cross-validation folds for every species.

    This helper mirrors :func:`run_baseline_all_species` but relies on the
    ``fold`` column of the training data instead of sequential holdouts.
    It calls :func:`run_fold_cv` for each species and writes the results to
    ``exp_dir/<species>``.

    Parameters
    ----------
    exp_dir : Path
        Directory where per-species results will be stored.
    best_k : int
        Default number of features used for the otter dataset and as fallback
        for other species.
    cutoff : dict[str, Any]
        Mapping of species names to optional ``"k"`` and ``"ward"`` overrides.
    subsample : bool, optional
        When ``True`` a subset of trail pairs is sampled for each fold via
        :func:`generate_pairwise_comparisons_from_df`.
    reuse_summary : bool, optional
        Skip processing when ``exp_dir/<species>/summary.csv`` already exists.
    n_jobs : int, optional
        Parallel jobs forwarded to :func:`run_fold_cv`.  ``-1`` uses all cores.
    use_sex_predictions : bool, optional
        When ``True`` sex-model probabilities are loaded via
        :func:`load_sex_predictions` and appended before evaluation.
    use_sexmodel_prediction : bool, optional
        Forwarded to :func:`run_fold_cv` to toggle usage of sex-model
        predictions inside the pairwise pipeline.
    sexmodel_path : str or Path, optional
        Path to a saved sex model forwarded to :func:`run_fold_cv` when
        ``use_sexmodel_prediction`` is ``True``.  When ``None`` the path is
        resolved automatically for each species using
        ``PATHS['random_search']/best_balanced_test_acc/<species>.joblib``.
    models_dir : Path, optional
        Directory containing the saved sex models used by
        :func:`load_sex_predictions`.
    selection_method, reducers, n_components, scaler_methods : optional
        Parameters forwarded to :func:`run_fold_cv` controlling feature
        selection, dimensionality reduction and scaling.
    """

    exp_dir = Path(exp_dir)
    exp_dir.mkdir(parents=True, exist_ok=True)

    for species_dir in tqdm(sorted(SPLITS_DIR.iterdir()), desc="Species"):
        if not species_dir.is_dir():
            continue

        out_dir = exp_dir / species_dir.name
        summary_fp = out_dir / "summary.csv"
        if reuse_summary and summary_fp.exists():
            continue

        try:
            df = _load_splits(species_dir, include_test=False)
        except FileNotFoundError:
            warnings.warn(
                f"Split directory for {species_dir.name!r} not found \u2013 skipping.",
                UserWarning,
            )
            continue

        feature_cols = get_feature_cols(df)

        if species_dir.name == "eurasian_otter":
            k = best_k
            ward = None
        else:
            spec_cfg = cutoff.get(species_dir.name, {})
            k = spec_cfg.get("k", best_k)
            ward = spec_cfg.get("ward")

        preds = None
        if use_sex_predictions:
            preds = load_sex_predictions(
                species_dir.name,
                models_dir=models_dir,
            )

        model_fp = sexmodel_path
        if use_sexmodel_prediction and model_fp is None:
            model_fp = (
                PATHS["random_search"]
                / "best_balanced_test_acc"
                / f"{species_dir.name}.joblib"
            )

        kwargs = dict(
            sex_predictions=preds,
            out_dir=out_dir,
            k_features=k,
            trail_col="Trail",
            subsample=subsample,
            cutoff=ward,
            reuse_summary=reuse_summary,
            n_jobs=n_jobs,
            use_sexmodel_prediction=use_sexmodel_prediction,
            sexmodel_path=model_fp,
            selection_method=selection_method,
            reducers=reducers,
            scaler_methods=scaler_methods,
        )
        if n_components is not None:
            kwargs["n_components"] = n_components

        run_fold_cv(
            df,
            feature_cols,
            **kwargs,
        )


def run_sex_prediction_experiment(
    exp_dir: Path,
    best_k: int,
    cutoff: Dict[str, Any],
    *,
    models_dir: str | Path | None = None,
    reuse_results: bool = True,
) -> None:
    """Evaluate sequential holdouts with and without sex predictions.

    For each species two runs of :func:`sequential_holdout.run` are performed:
    one with appended sex-model probabilities and one without.  The resulting
    summaries are written to ``exp_dir/<species>/with_sex`` and
    ``exp_dir/<species>/without_sex`` respectively.

    Parameters
    ----------
    exp_dir : Path
        Root directory for the output structure.
    best_k : int
        Default number of features for all species.
    cutoff : dict[str, Any]
        Mapping of species specific Ward cut-offs and feature counts.
    models_dir : str or Path, optional
        Directory containing saved sex models loaded by
        :func:`load_sex_predictions`.
    reuse_results : bool, optional
        When ``True`` (default) existing ``summary.csv`` files are loaded and
        the computation for that setup is skipped.
    """

    exp_dir = Path(exp_dir)
    exp_dir.mkdir(parents=True, exist_ok=True)

    for species_dir in tqdm(sorted(SPLITS_DIR.iterdir()), desc="Species"):
        if not species_dir.is_dir():
            continue

        try:
            df = _load_splits(species_dir)
        except FileNotFoundError:
            warnings.warn(
                f"Split directory for {species_dir.name!r} not found \u2013 skipping.",
                UserWarning,
            )
            continue
        feature_cols = get_feature_cols(df)

        spec_cfg = cutoff.get(species_dir.name, {})
        k = spec_cfg.get("k", best_k)
        ward = spec_cfg.get("ward")

        preds = load_sex_predictions(
            species_dir.name,
            models_dir=models_dir,
        )

        out_with = exp_dir / species_dir.name / "with_sex"
        with_sum = out_with / "summary.csv"
        if not (reuse_results and with_sum.exists()):
            sequential_holdout.run(
                df,
                feature_cols,
                sex_predictions=preds,
                iterations=1,
                out_dir=out_with,
                k_features=k,
                trail_col="Trail",
                subsample=False,
                cutoff=ward,
                reuse_summary=reuse_results,
            )

        out_without = exp_dir / species_dir.name / "without_sex"
        without_sum = out_without / "summary.csv"
        if not (reuse_results and without_sum.exists()):
            sequential_holdout.run(
                df,
                feature_cols,
                iterations=1,
                out_dir=out_without,
                k_features=k,
                trail_col="Trail",
                subsample=False,
                cutoff=ward,
                reuse_summary=reuse_results,
            )


def _load_splits(species_dir: Path, *, include_test: bool = True) -> pd.DataFrame:
    """Return concatenated train and optionally test tables for ``species_dir``.

    Parameters
    ----------
    species_dir:
        Path pointing to the directory containing ``train.parquet`` and
        optionally ``test.parquet`` files.
    include_test:
        When ``True`` (default) the ``test.parquet`` file is loaded as well.
        Set to ``False`` to work exclusively with the training split.
    """

    train_fp = species_dir / "train.parquet"
    test_fp = species_dir / "test.parquet"
    parts = []
    if train_fp.exists():
        parts.append(pd.read_parquet(train_fp))
    if include_test and test_fp.exists():
        parts.append(pd.read_parquet(test_fp))
    if not parts:
        raise FileNotFoundError(f"No train/test splits found in {species_dir}")

    df = pd.concat(parts, ignore_index=True)

    if "Fold" in df.columns and "fold" not in df.columns:
        df = df.rename(columns={"Fold": "fold"})

    if "trail" in df.columns and "Trail" not in df.columns:
        df = df.rename(columns={"trail": "Trail"})

    if "Trail" in df.columns:
        df = df.dropna(subset=["Trail"])
        df = df[df["Trail"].astype(str).str.lower() != "unknown"]

    return df


def load_sex_predictions(
    species: str,
    *,
    prefer_generic: bool = True,
    models_dir: str | Path | None = None,
) -> pd.DataFrame:
    """Return sex-model predictions for ``species``.

    This is a thin wrapper around :func:`predict_all` from the sex classification
    pipeline.  The helper simply forwards the parameters and returns the
    resulting DataFrame so that the individual ID baseline can load the
    predictions without importing the full sex pipeline here.
    """

    from FIT_python.pipeline_sex.sex_predict_and_visualisation import predict_all

    return predict_all(
        species,
        prefer_generic=prefer_generic,
        models_dir=models_dir,
    )


def add_sex_features(df: pd.DataFrame, pred_df: pd.DataFrame) -> pd.DataFrame:
    """Append sex probabilities to ``df`` and compute trail-level aggregates."""

    if "id" not in df.columns:
        raise KeyError("DataFrame must contain an 'id' column")

    prob_cols = [
        c for c in pred_df.columns if c.startswith("pred_") and c.endswith("proba_f")
    ]
    if not prob_cols:
        return df

    pred_sub = pred_df.set_index("id")[prob_cols]
    out = df.set_index("id").join(pred_sub, how="left").reset_index()

    if "Trail" in out.columns:
        out["trail_proba_f"] = (
            out.groupby("Trail")[prob_cols].transform("mean").mean(axis=1)
        )

    return out


def run_fold_cv(
    df: pd.DataFrame,
    feature_cols: Iterable[str],
    sex_predictions: pd.DataFrame | None = None,
    *,
    sample_col: str = "id",
    id_col: str = "individual_id",
    fold_col: str = "fold",
    out_dir: Path | None = None,
    n_jobs: int = -1,
    reuse_summary: bool = True,
    k_features: int | None = None,
    trail_col: str = "trail",
    subsample: bool = False,
    cutoff: float | None = None,
    overlap_prob: float = 0.5,
    selection_method: str = SOFT_CONFIG["pipeline_individual_id"]["pairwise_defaults"][
        "selection_method"
    ],
    reducers: Iterable[str] | str | None = SOFT_CONFIG["pipeline_individual_id"][
        "pairwise_defaults"
    ]["reducers"],
    n_components: int | Iterable[int] = SOFT_CONFIG["pipeline_individual_id"][
        "pairwise_defaults"
    ]["n_components"],
    outlier_methods: Iterable[str]
    | str
    | None = SOFT_CONFIG["pipeline_individual_id"]["pairwise_defaults"][
        "outlier_methods"
    ],
    scaler_methods: Iterable[str]
    | str
    | None = SOFT_CONFIG["pipeline_individual_id"]["pairwise_defaults"][
        "scaler_methods"
    ],
    use_sexmodel_prediction: bool = SOFT_CONFIG["pipeline_individual_id"][
        "pairwise_defaults"
    ]["use_sexmodel_prediction"],
    sexmodel_path: str | None = None,
) -> pd.DataFrame:
    """Evaluate pairwise pipeline using predefined folds.

    The function iterates over unique values in ``fold_col`` and treats each
    fold as validation set while the remaining data forms the training set.  The
    results for every fold are written to ``out_dir`` as ``fold_<n>.csv`` with a
    combined ``summary.csv`` containing the evaluation metrics.  Summary rows
    now also include a ``pipeline`` identifier describing the preprocessing and
    reduction steps chosen inside
    :func:`run_all_pairwise_projections_parallel`.

    Parameters
    ----------
    selection_method, reducers, n_components, scaler_methods
        Parameters forwarded to ``run_all_pairwise_projections_parallel`` to
        control feature selection, dimensionality reduction and scaling.
    sexmodel_path : str, optional
        Path to a saved sex classifier.  When ``use_sexmodel_prediction`` is
        ``True`` and no path is given, the function attempts to resolve the
        model path from the ``species`` column using
        ``PATHS['random_search']/best_balanced_test_acc/<species>.joblib``.
    """

    from .generate_trails_and_trailpairs import generate_pairwise_comparisons_from_df
    from .pairwise_individual_id_pipeline import run_all_pairwise_projections_parallel
    from .evaluation import (
        compute_confusion,
        compute_overlap_jsl_style_vec,
        compute_bcr,
    )
    from .population_estimation import (
        cluster_population,
        compute_erd,
        optimal_cutoff,
        concordance_correlation_coefficient,
    )
    import numpy as np

    out_dir = Path(out_dir or RESULTS_DATA_DIR / "individual_id")
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_fp = out_dir / "summary.csv"
    if reuse_summary and summary_fp.exists():
        return pd.read_csv(summary_fp)

    df_all, pred_cols = sequential_holdout._merge_predictions(
        df, sex_predictions, sample_col=sample_col
    )
    use_cols = list(feature_cols) + pred_cols

    model_fp = sexmodel_path
    if use_sexmodel_prediction and model_fp is None:
        if "species" not in df_all.columns:
            raise ValueError(
                "sexmodel_path must be provided when use_sexmodel_prediction=True"
            )
        species = str(df_all["species"].dropna().unique()[0])
        model_fp = (
            PATHS["random_search"]
            / "best_balanced_test_acc"
            / f"{species}.joblib"
        )

    if fold_col not in df_all.columns:
        raise KeyError(f"DataFrame must contain '{fold_col}' column")

    folds = sorted(df_all[fold_col].dropna().unique())

    all_parts: list[pd.DataFrame] = []
    summaries: list[dict[str, Any]] = []

    for fold in folds:
        df_train = df_all[df_all[fold_col] != fold]
        df_val = df_all[df_all[fold_col] == fold]

        comps, _ = generate_pairwise_comparisons_from_df(
            df_val,
            trail_col=trail_col,
            subsample=subsample,
            fold_col=fold_col,
        )
        if not comps:
            continue

        base_df = pd.concat([df_train, df_val], ignore_index=True)
        kwargs = {
            "n_jobs": n_jobs,
            "feature_cols": use_cols,
            "selection_method": selection_method,
            "reducers": reducers,
            "outlier_methods": outlier_methods,
            "scaler_methods": scaler_methods,
            "use_sexmodel_prediction": use_sexmodel_prediction,
            "sexmodel_path": model_fp,
        }
        if n_components is not None:
            kwargs["n_components"] = n_components
        if k_features is not None:
            kwargs["k_features"] = k_features
        res = run_all_pairwise_projections_parallel(
            comps,
            base_df,
            fit_df=df_train,
            **kwargs,
        )
        df_res = pd.DataFrame(res)
        if df_res.empty:
            continue

        df_res["fold"] = fold
        all_parts.append(df_res.copy())

        pipeline_name = (
            df_res["pipeline"].iloc[0] if "pipeline" in df_res.columns else ""
        )

        df_res["pred"] = compute_overlap_jsl_style_vec(df_res, p=overlap_prob)
        cm = compute_confusion(df_res, true_col="same_individual", pred_col="pred")
        bcr = compute_bcr(cm)

        arr, trails = _pairs_to_array(
            df_res["trail_a_id"], df_res["trail_b_id"], df_res["dist_euclidean"]
        )
        dist_mat = pd.DataFrame(arr, index=trails, columns=trails)
        dist_mat = dist_mat.fillna(np.nanmax(arr))

        true_n = df_val[id_col].dropna().astype(str).nunique()
        if cutoff is None:
            ward_cutoff, ci = optimal_cutoff(dist_mat, true_n)
            cutoff_low, cutoff_high = ci
        else:
            ward_cutoff = float(cutoff)
            cutoff_low = ward_cutoff
            cutoff_high = ward_cutoff

        pred_n = cluster_population(dist_mat, ward_cutoff)
        erd = compute_erd(pred_n, true_n)

        summaries.append(
            {
                "pipeline": pipeline_name,
                "fold": fold,
                "bcr": bcr,
                "pred_count": pred_n,
                "true_count": true_n,
                "erd": erd,
                "ward_cutoff": ward_cutoff,
                "cutoff_low": cutoff_low,
                "cutoff_high": cutoff_high,
                "tp": int(cm.loc["true_same", "pred_same"]),
                "fp": int(cm.loc["true_diff", "pred_same"]),
                "tn": int(cm.loc["true_diff", "pred_diff"]),
                "fn": int(cm.loc["true_same", "pred_diff"]),
                "n_pairs": len(df_res),
            }
        )

        df_res.to_csv(out_dir / f"fold_{fold}.csv", index=False)

    if all_parts:
        df_all_pairs = pd.concat(all_parts, ignore_index=True)
        df_all_pairs.to_csv(out_dir / "all_folds.csv", index=False)

    summary_df = pd.DataFrame(summaries)
    if not summary_df.empty:
        ccc = concordance_correlation_coefficient(
            summary_df["pred_count"], summary_df["true_count"]
        )
        summary_df["ccc"] = ccc
    else:
        summary_df["ccc"] = float("nan")
    summary_df.to_csv(summary_fp, index=False)
    return summary_df
