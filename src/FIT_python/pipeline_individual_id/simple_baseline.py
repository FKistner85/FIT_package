from __future__ import annotations


"""Utility helpers for a simple ID baseline."""

from pathlib import Path
from typing import Iterable, List, Dict, Any
from tqdm.auto import tqdm

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from FIT_python.caption_utils import save_caption
from FIT_python.Visualisations.plot_style import apply_style
from .population_estimation import concordance_correlation_coefficient
from FIT_python.config import SPLITS_DIR
from FIT_python.data_split_and_summary.data_import_utils import get_feature_cols
from . import sequential_holdout
from FIT_python.pipeline_sex.sex_predict_and_visualisation import predict_all


def load_sex_predictions(species: str) -> pd.DataFrame:
    """Return sex-model predictions for ``species``."""

    return predict_all(species, prefer_generic=True)


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
        records.append({"species": csv.parent.name, "bcr": mean_bcr, "erd": mean_erd, "ccc": ccc})

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
    sns.barplot(data=df, x="species", y="bcr", order=order, ax=ax, color="#4C72B0", edgecolor="black")
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
) -> int:
    """Run sequential holdouts for the otter data across ``k_range`` values."""

    exp_dir = Path(exp_dir)
    out_dir = exp_dir / "otter"
    out_dir.mkdir(parents=True, exist_ok=True)

    species_dir = SPLITS_DIR / "eurasian_otter"
    df = _load_splits(species_dir)
    feature_cols = get_feature_cols(df)

    sex_preds = load_sex_predictions(species_dir.name) if use_sex_predictions else None

    best_k: int | None = None
    best_bcr = float("-inf")

    best_summary: pd.DataFrame | None = None

    for k in k_range:
        k_dir = out_dir / f"k{k}"
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
        pd.DataFrame([agg]).to_csv(out_dir / f"summary_k{k}.csv", index=False)

        mean_bcr = float(agg.get("bcr", float("nan")))
        if mean_bcr > best_bcr:
            best_bcr = mean_bcr
            best_k = k
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

    Parameters
    ----------
    exp_dir:
        Root directory for all results. One subdirectory per species will be
        created underneath this path.
    best_k:
        Number of features for the otter experiments and the default for the
        benchmark species.
    cutoff:
        Mapping of species name to parameter dictionary with keys ``k`` and
        ``ward`` specifying deviations from ``best_k`` and the Ward cut-off
        distance.
    reuse_summary:
        When ``True`` and ``exp_dir/<species>/summary.csv`` exists, the
        computation for that species is skipped.
    n_jobs:
        Parallel jobs forwarded to :func:`sequential_holdout.run`. ``-1`` uses
        all available CPU cores.
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

        df = _load_splits(species_dir)
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


def run_sex_prediction_experiment(exp_dir: Path, best_k: int, cutoff: Dict[str, Any]) -> None:
    """Evaluate sequential holdouts with and without sex predictions."""

    exp_dir = Path(exp_dir)
    exp_dir.mkdir(parents=True, exist_ok=True)

    for species_dir in tqdm(sorted(SPLITS_DIR.iterdir()), desc="Species"):
        if not species_dir.is_dir():
            continue

        df = _load_splits(species_dir)
        feature_cols = get_feature_cols(df)

        spec_cfg = cutoff.get(species_dir.name, {})
        k = spec_cfg.get("k", best_k)
        ward = spec_cfg.get("ward")

        preds = load_sex_predictions(species_dir.name)

        out_with = exp_dir / species_dir.name / "with_sex"
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
        )

        out_without = exp_dir / species_dir.name / "without_sex"
        sequential_holdout.run(
            df,
            feature_cols,
            iterations=1,
            out_dir=out_without,
            k_features=k,
            trail_col="Trail",
            subsample=False,
            cutoff=ward,
        )


def _load_splits(species_dir: Path) -> pd.DataFrame:
    """Return concatenated train and test tables for ``species_dir``."""

    train_fp = species_dir / "train.parquet"
    test_fp = species_dir / "test.parquet"
    parts = []
    if train_fp.exists():
        parts.append(pd.read_parquet(train_fp))
    if test_fp.exists():
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


def load_sex_predictions(species: str, prefer_generic: bool = True) -> pd.DataFrame:
    """Return sex-model predictions for ``species``.

    This is a thin wrapper around :func:`predict_all` from the sex
    classification pipeline.  The helper simply forwards the parameters and
    returns the resulting DataFrame so that the individual ID baseline can load
    the predictions without importing the full sex pipeline here.
    """

    from FIT_python.pipeline_sex.sex_predict_and_visualisation import predict_all

    return predict_all(species, prefer_generic=prefer_generic)


def add_sex_features(df: pd.DataFrame, pred_df: pd.DataFrame) -> pd.DataFrame:
    """Append sex probabilities to ``df`` and compute trail-level aggregates."""

    if "id" not in df.columns:
        raise KeyError("DataFrame must contain an 'id' column")

    prob_cols = [c for c in pred_df.columns if c.startswith("pred_") and c.endswith("proba_f")]
    if not prob_cols:
        return df

    pred_sub = pred_df.set_index("id")[prob_cols]
    out = df.set_index("id").join(pred_sub, how="left").reset_index()

    if "Trail" in out.columns:
        out["trail_proba_f"] = (
            out.groupby("Trail")[prob_cols].transform("mean").mean(axis=1)
        )

    return out


