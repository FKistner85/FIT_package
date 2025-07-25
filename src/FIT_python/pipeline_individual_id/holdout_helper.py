from __future__ import annotations

"""Helpers to generate sequential holdout sets from split directories."""

from pathlib import Path
from typing import Sequence, Dict, List

from FIT_python.config import GLOBAL_RANDOM_SEED

import pandas as pd

from .evaluation import sequential_holdout_ids


def generate_holdout_sets(
    split_dir: Path,
    val_sizes: Sequence[int],
    iterations: int,
    seed: int = GLOBAL_RANDOM_SEED,
) -> dict[str, list[dict[str, pd.DataFrame]]]:
    """Return sequential holdout splits for every species.

    Parameters
    ----------
    split_dir:
        Directory containing per-species ``train.parquet`` files.
    val_sizes:
        Validation set sizes passed to :func:`sequential_holdout_ids`.
    iterations:
        Number of sequential holdout iterations.
    seed:
        Random seed for reproducible splits. Defaults to ``GLOBAL_RANDOM_SEED``.

    Returns
    -------
    dict[str, list[dict[str, pd.DataFrame]]]
        Mapping from species codes to lists of split dictionaries. Each
        dictionary contains ``train_df`` and ``val_df`` along with
        ``iteration`` and ``n_val`` information.
    """
    split_dir = Path(split_dir)
    result: Dict[str, List[Dict[str, pd.DataFrame]]] = {}
    for species_dir in split_dir.iterdir():
        if not species_dir.is_dir():
            continue
        train_fp = species_dir / "train.parquet"
        if not train_fp.exists():
            continue
        df = pd.read_parquet(train_fp)
        unique_ids = df["individual_id"].dropna().astype(str).unique()
        splits = sequential_holdout_ids(
            unique_ids, val_sizes=val_sizes, n_iter=iterations, random_state=seed
        )
        species_splits: List[Dict[str, pd.DataFrame]] = []
        for split in splits:
            train_ids = set(split["train_ids"])
            val_ids = set(split["val_ids"])
            df_train = df[df["individual_id"].astype(str).isin(train_ids)]
            df_val = df[df["individual_id"].astype(str).isin(val_ids)]
            species_splits.append(
                {
                    "train_df": df_train.reset_index(drop=True),
                    "val_df": df_val.reset_index(drop=True),
                    "iteration": split["iteration"],
                    "n_val": split["n_val"],
                }
            )
        result[species_dir.name] = species_splits
    return result
