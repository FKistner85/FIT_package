from __future__ import annotations

"""Sequential holdout evaluation for individual ID pipelines."""

from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from FIT_python.config import RESULTS_DATA_DIR
from FIT_python.soft_config import SOFT_CONFIG

from .evaluation import (
    sequential_holdout_ids,
    compute_confusion,
    compute_overlap_jsl_style,
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


def _merge_predictions(df: pd.DataFrame, preds: pd.DataFrame | None, *, sample_col: str = "id") -> tuple[pd.DataFrame, list[str]]:
    """Return ``df`` merged with ``preds`` and list of added columns."""
    if preds is None:
        return df, []

    df = df.copy()
    if sample_col in preds.columns:
        preds = preds.set_index(sample_col)
    df = df.set_index(sample_col)
    df = df.join(preds, how="left")
    df = df.reset_index()
    pred_cols = [c for c in preds.columns]
    return df, pred_cols


def run(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    sex_predictions: pd.DataFrame | None = None,
    *,
    sample_col: str = "id",
    id_col: str = "individual_id",
    iterations: int = 1,
    val_sizes: Iterable[int] | None = None,
    random_state: int | None = None,
    out_dir: Path | None = None,
    n_jobs: int = -1,
    reuse_summary: bool = True,
    k_features: int | None = None,
    trail_col: str = "trail",
    subsample: bool = False,
    cutoff: float | None = None,
    overlap_prob: float = 0.5,
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
        Random seed for the split generator.
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
        Probability mass for :func:`compute_overlap_jsl_style` when predicting
        whether two trails belong to the same individual.

    Returns
    -------
    pd.DataFrame
        Summary table with one row per split containing the BCR. In addition
        to the per-split ``split_*.csv`` files, a combined ``all_splits.csv``
        containing all pairwise results is written to ``out_dir``.
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
        kwargs = {"n_jobs": n_jobs, "feature_cols": use_cols}
        if k_features is not None:
            kwargs["k_features"] = k_features
        res = run_all_pairwise_projections_parallel(comps, base_df, **kwargs)
        df_res = pd.DataFrame(res)
        if df_res.empty:
            continue

        df_res["split"] = idx
        df_res["iteration"] = split["iteration"]
        df_res["n_val"] = split["n_val"]
        all_parts.append(df_res.copy())

        df_res["pred"] = df_res.apply(
            compute_overlap_jsl_style, axis=1, p=overlap_prob
        )
        cm = compute_confusion(df_res, true_col="same_individual", pred_col="pred")
        bcr = compute_bcr(cm)

        # --- build square distance matrix ---
        trails = sorted(set(df_res["trail_a_id"]) | set(df_res["trail_b_id"]))
        dist_mat = pd.DataFrame(np.nan, index=trails, columns=trails)
        for a, b, val in zip(df_res["trail_a_id"], df_res["trail_b_id"], df_res["dist_euclidean"]):
            try:
                v = float(val)
            except Exception:
                continue
            dist_mat.at[a, b] = v
            dist_mat.at[b, a] = v
        np.fill_diagonal(dist_mat.values, 0.0)
        max_d = np.nanmax(dist_mat.values)
        dist_mat = dist_mat.fillna(max_d)

        true_n = len(val_ids)
        if cutoff is None:
            cut, _ = optimal_cutoff(dist_mat, true_n)
        else:
            cut = float(cutoff)
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
        trails = sorted(set(part["trail_a_id"]) | set(part["trail_b_id"]))
        dist_mat = pd.DataFrame(np.nan, index=trails, columns=trails)
        for a, b, val in zip(part["trail_a_id"], part["trail_b_id"], part["dist_euclidean"]):
            try:
                v = float(val)
            except Exception:
                continue
            dist_mat.at[a, b] = v
            dist_mat.at[b, a] = v

        np.fill_diagonal(dist_mat.values, 0.0)
        max_d = np.nanmax(dist_mat.values)
        dist_mat = dist_mat.fillna(max_d)

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
