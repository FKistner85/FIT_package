from __future__ import annotations

"""Convenience utilities for a lightweight individual ID baseline."""

from pathlib import Path
from typing import Iterable, Dict, Any

import pandas as pd

from FIT_python.config import SPLITS_DIR
from FIT_python.data_split_and_summary.data_import_utils import get_feature_cols

from . import sequential_holdout


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
    return pd.concat(parts, ignore_index=True)


def run_simple_baseline_otter(
    exp_dir: Path, k_range: Iterable[int] = range(12, 21), iterations: int = 10
) -> int:
    """Run sequential holdouts for the otter data across ``k_range`` values."""

    exp_dir = Path(exp_dir)
    out_dir = exp_dir / "otter"
    out_dir.mkdir(parents=True, exist_ok=True)

    species_dir = SPLITS_DIR / "eurasian_otter"
    df = _load_splits(species_dir)
    feature_cols = get_feature_cols(df)

    best_k: int | None = None
    best_bcr = float("-inf")

    for k in k_range:
        k_dir = out_dir / f"k{k}"
        summary = sequential_holdout.run(
            df,
            feature_cols,
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

    if best_k is None:
        raise RuntimeError("No valid results computed for otter baseline")
    return best_k


def run_baseline_all_species(exp_dir: Path, best_k: int, cutoff: Dict[str, Any]) -> None:
    """Evaluate the baseline for every species using sequential holdouts."""

    exp_dir = Path(exp_dir)
    exp_dir.mkdir(parents=True, exist_ok=True)

    for species_dir in sorted(SPLITS_DIR.iterdir()):
        if not species_dir.is_dir():
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

        out_dir = exp_dir / species_dir.name
        sequential_holdout.run(
            df,
            feature_cols,
            iterations=1,
            out_dir=out_dir,
            k_features=k,
            trail_col="Trail",
            subsample=False,
            cutoff=ward,
        )

