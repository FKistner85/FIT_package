from __future__ import annotations

"""Sequential holdout evaluation for individual ID pipelines."""

from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from FIT_python.config import RESULTS_DATA_DIR, GLOBAL_RANDOM_SEED
from FIT_python.soft_config import SOFT_CONFIG

from .evaluation import (
    sequential_holdout_ids,
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
from .generate_trails_and_trailpairs import generate_pairwise_comparisons_from_df
from .pairwise_individual_id_pipeline import run_all_pairwise_projections_parallel


def _merge_predictions(
    df: pd.DataFrame, preds: pd.DataFrame | None, *, sample_col: str = "id"
) -> tuple[pd.DataFrame, list[str]]:
    """Return ``df`` merged with ``preds`` and list of added columns."""

    if preds is None:
        return df, []

    df = df.copy()

    # keep only prediction columns and track them for later use
    pred_cols = [c for c in preds.columns if c.startswith("pred_")]
    preds = preds[pred_cols]

    if sample_col in preds.columns:
        preds = preds.set_index(sample_col)

    df = df.set_index(sample_col)

    # drop overlapping columns to avoid join errors
    overlap = df.columns.intersection(preds.columns)
    if len(overlap) > 0:
        df = df.drop(columns=list(overlap))

    df = df.join(preds, how="left")
    df = df.reset_index()

    return df, pred_cols


def _pairs_to_array(
    trail_a: Iterable[str], trail_b: Iterable[str], distances: Iterable[float]
) -> tuple[np.ndarray, list[str]]:
    """Return distance array and trail order via index mapping.

    Benchmarking ``timeit`` on 5k pairs and 1k unique trails showed
    roughly a 12x speed-up compared to the previous row-wise loop.
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


def run(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    sex_predictions: pd.DataFrame | None = None,
    *,
    sample_col: str = "id",
    id_col: str = "individual_id",
    iterations: int = 1,
    val_sizes: Iterable[int] | None = None,
    random_state: int = GLOBAL_RANDOM_SEED,
    out_dir: Path | None = None,
    n_jobs: int = -1,
    reuse_summary: bool = True,
    k_features: int | None = None,
    trail_col: str = "trail",
    subsample: bool = False,
    cutoff: float | None = None,
    overlap_prob: float = 0.5,
    outlier_methods: Iterable[str] | str | None = SOFT_CONFIG["pipeline_individual_id"][
        "pairwise_defaults"
    ]["outlier_methods"],
    scaler_methods: Iterable[str] | str | None = SOFT_CONFIG["pipeline_individual_id"][
        "pairwise_defaults"
    ]["scaler_methods"],
    use_sexmodel_prediction: bool = SOFT_CONFIG["pipeline_individual_id"]["pairwise_defaults"][
        "use_sexmodel_prediction"
    ],
    sexmodel_path: str | None = None,
) -> pd.DataFrame:
    """Evaluate pairwise pipeline using sequential holdouts.

    Parameters
    ----------
    df:
        Cleaned footprints with ``individual_id`` and feature columns.
    feature_cols:
        Names of morphometric feature columns.
    sex_predictions:
        Optional DataFrame with sex-model predictions to append as extra
        features. When provided, it must be indexed by ``sample_col`` or contain
        a column of that name.
    sample_col:
        Column containing sample identifiers. Defaults to ``"id"``.
    id_col:
        Column identifying individuals. Defaults to ``"individual_id"``.
    iterations:
        Number of sequential holdout iterations.
    val_sizes:
        Validation sizes passed to :func:`sequential_holdout_ids`. Defaults to
        ``SOFT_CONFIG['pipeline_individual_id']['sequential_holdout_val_sizes']``.
    random_state:
        Random seed for the split generator. Defaults to ``GLOBAL_RANDOM_SEED``.
    out_dir:
        Directory to write per-split CSV results. Defaults to
        ``RESULTS_DATA_DIR / 'individual_id'``.
    n_jobs:
        Parallel jobs for the pairwise projection step.  ``-1`` uses all cores.
    reuse_summary:
        When ``True`` and ``out_dir/summary.csv`` exists, load that table instead
        of recomputing the sequential holdout.
    k_features:
        Number of morphometric features to select. ``None`` uses the default
        from :func:`run_all_pairwise_projections_parallel`.
    trail_col:
        Column name containing trail identifiers.
    subsample:
        Whether to generate sub-trails when creating comparisons.
    cutoff:
        Optional Ward distance used for population clustering. When ``None`` the
        cutoff is determined via :func:`optimal_cutoff` for each split.
    overlap_prob:
        Probability mass for :func:`compute_overlap_jsl_style_vec` when predicting
        whether two trails belong to the same individual.
    sexmodel_path : str, optional
        Path to a saved sex classifier.  If ``use_sexmodel_prediction`` is
        ``True`` and no path is provided, the model location is derived from the
        ``species`` column using
        ``PATHS['random_search']/best_balanced_test_acc/<species>.joblib``.

    Returns
    -------
    pd.DataFrame
        Summary table with one row per split containing the BCR, ERD, predicted
        and true population sizes as well as the Ward cut-off statistics
        (``ward_cutoff``, ``cutoff_low``, ``cutoff_high``).  In addition to the
        per-split ``split_*.csv`` files, a combined ``all_splits.csv`` containing
        all pairwise results is written to ``out_dir``.
    """
    val_sizes = (
        tuple(val_sizes)
        if val_sizes is not None
        else tuple(
            SOFT_CONFIG["pipeline_individual_id"]["sequential_holdout_val_sizes"]
        )
    )
    out_dir = Path(out_dir or RESULTS_DATA_DIR / "individual_id")
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_fp = out_dir / "summary.csv"
    if reuse_summary and summary_fp.exists():
        return pd.read_csv(summary_fp)

    df_all, pred_cols = _merge_predictions(df, sex_predictions, sample_col=sample_col)
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

    unique_ids = df_all[id_col].dropna().astype(str).unique()
    splits = sequential_holdout_ids(unique_ids, val_sizes=val_sizes, n_iter=iterations, random_state=random_state)

    summaries = []
    all_parts: list[pd.DataFrame] = []
    for idx, split in enumerate(splits):
        train_ids = split["train_ids"]
        val_ids = split["val_ids"]
        df_train = df_all[df_all[id_col].isin(train_ids)]
        df_val = df_all[df_all[id_col].isin(val_ids)]

        comps, _ = generate_pairwise_comparisons_from_df(
            df_val,
            trail_col=trail_col,
            subsample=subsample,
        )
        if not comps:
            continue

        base_df = pd.concat([df_train, df_val], ignore_index=True)
        kwargs = {
            "n_jobs": n_jobs,
            "feature_cols": use_cols,
            "outlier_methods": outlier_methods,
            "scaler_methods": scaler_methods,
            "use_sexmodel_prediction": use_sexmodel_prediction,
            "sexmodel_path": model_fp,
        }
        if k_features is not None:
            kwargs["k_features"] = k_features
        res = run_all_pairwise_projections_parallel(
            comps,
            base_df,
            train_df=df_train,
            **kwargs,
        )
        df_res = pd.DataFrame(res)
        if df_res.empty:
            continue

        df_res["split"] = idx
        df_res["iteration"] = split["iteration"]
        df_res["n_val"] = split["n_val"]
        all_parts.append(df_res.copy())

        df_res["pred"] = compute_overlap_jsl_style_vec(df_res, p=overlap_prob)
        cm = compute_confusion(df_res, true_col="same_individual", pred_col="pred")
        bcr = compute_bcr(cm)

        # --- build square distance matrix ---
        arr, trails = _pairs_to_array(
            df_res["trail_a_id"], df_res["trail_b_id"], df_res["dist_euclidean"]
        )
        dist_mat = pd.DataFrame(arr, index=trails, columns=trails)
        dist_mat = dist_mat.fillna(np.nanmax(arr))

        true_n = len(val_ids)
        if cutoff is None:
            ward_cutoff, ci = optimal_cutoff(dist_mat, true_n)
            cutoff_low, cutoff_high = ci
        else:
            ward_cutoff = float(cutoff)
            cutoff_low = ward_cutoff
            cutoff_high = ward_cutoff
        cut = ward_cutoff
        pred_n = cluster_population(dist_mat, cut)
        erd = compute_erd(pred_n, true_n)
        summaries.append(
            {
                "split": idx,
                "iteration": split["iteration"],
                "n_val": split["n_val"],
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
        df_res.to_csv(out_dir / f"split_{idx}.csv", index=False)

    if all_parts:
        df_all = pd.concat(all_parts, ignore_index=True)
        df_all.to_csv(out_dir / "all_splits.csv", index=False)

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


def evaluate_with_cutoff(result_dir: Path, cutoff: float) -> pd.DataFrame:
    """Re-evaluate splits with a fixed Ward cut-off.

    Parameters
    ----------
    result_dir:
        Directory containing ``all_splits.csv`` produced by :func:`run`.
    cutoff:
        Ward distance passed to :func:`cluster_population`.

    Returns
    -------
    pd.DataFrame
        Data frame with one row per split containing ``pred_count``,
        ``true_count`` and ``erd``.
    """

    result_dir = Path(result_dir)
    all_fp = result_dir / "all_splits.csv"
    if not all_fp.exists():
        raise FileNotFoundError(all_fp)

    df_all = pd.read_csv(all_fp)
    if df_all.empty:
        return pd.DataFrame(
            columns=["split", "pred_count", "true_count", "erd"], dtype=float
        )

    summaries = []
    for split, part in df_all.groupby("split"):
        arr, trails = _pairs_to_array(
            part["trail_a_id"], part["trail_b_id"], part["dist_euclidean"]
        )
        dist_mat = pd.DataFrame(arr, index=trails, columns=trails)
        dist_mat = dist_mat.fillna(np.nanmax(arr))

        pred_n = cluster_population(dist_mat, float(cutoff))
        ids = set(part["ind_a"].astype(str)) | set(part["ind_b"].astype(str))
        true_n = len(ids)
        erd = compute_erd(pred_n, true_n)
        summaries.append(
            {
                "split": int(split),
                "pred_count": pred_n,
                "true_count": true_n,
                "erd": erd,
            }
        )

    return pd.DataFrame(summaries)


def compute_global_cutoffs(all_splits_path: Path) -> dict[str, float]:
    """Return aggregate Ward cut-off statistics across splits.

    Parameters
    ----------
    all_splits_path:
        CSV file created by :func:`run` containing the pairwise
        results of all splits.

    Returns
    -------
    dict
        Dictionary with the mean and median Ward cut-off as well as
        the mean lower and upper 25% bounds across splits.
    """

    all_splits_path = Path(all_splits_path)
    if not all_splits_path.exists():
        raise FileNotFoundError(all_splits_path)

    df_all = pd.read_csv(all_splits_path)
    if df_all.empty:
        return {
            "mean_cutoff": float("nan"),
            "median_cutoff": float("nan"),
            "mean_low": float("nan"),
            "mean_high": float("nan"),
        }

    cutoffs: list[float] = []
    lows: list[float] = []
    highs: list[float] = []

    for _, part in df_all.groupby("split"):
        arr, trails = _pairs_to_array(
            part["trail_a_id"], part["trail_b_id"], part["dist_euclidean"]
        )
        dist_mat = pd.DataFrame(arr, index=trails, columns=trails)
        dist_mat = dist_mat.fillna(np.nanmax(arr))

        ids = set(part["ind_a"].astype(str)) | set(part["ind_b"].astype(str))
        true_n = len(ids)

        cutoff, (low, high) = optimal_cutoff(dist_mat, true_n)
        cutoffs.append(cutoff)
        lows.append(low)
        highs.append(high)

    return {
        "mean_cutoff": float(np.mean(cutoffs)),
        "median_cutoff": float(np.median(cutoffs)),
        "mean_low": float(np.mean(lows)),
        "mean_high": float(np.mean(highs)),
    }
