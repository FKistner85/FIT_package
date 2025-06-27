from __future__ import annotations

"""Utilities for train/test and fold creation used in the pipeline."""

from pathlib import Path
import numpy as np
import pandas as pd
from typing import Tuple

from sklearn.model_selection import GroupKFold, KFold

from FIT_python.config import (
    SPLITS_DIR,
    GLOBAL_RANDOM_SEED,
    NUM_FOLDS,
    GROUP_COL,
)
from FIT_python.old_files.split_utils import group_stratified_kfold


__all__ = ["splits_available", "_make_folds"]


def splits_available() -> bool:
    """Return True if at least one train/test split pair exists."""
    if not Path(SPLITS_DIR).exists():
        return False
    for d in Path(SPLITS_DIR).iterdir():
        if not d.is_dir():
            continue
        if (d / "train.parquet").exists() and (d / "test.parquet").exists():
            return True
    return False


def _check_valid(fold_ids: np.ndarray, y: pd.Series, n_splits: int) -> bool:
    """Verify that each fold contains both classes 0 and 1."""
    for fold_i in range(n_splits):
        classes = set(y[fold_ids == fold_i].unique())
        if classes != {0, 1}:
            return False
    return True


def _make_folds(
    df: pd.DataFrame,
    y: pd.Series,
    n_splits: int = NUM_FOLDS,
    group_col: str = GROUP_COL,
) -> Tuple[np.ndarray, str]:
    """Generate fold ids using several fallbacks.

    Order of attempts:
      1. Existing ``Fold`` column ("predefined")
      2. ``group_stratified_kfold`` helper
      3. ``GroupKFold``
      4. ``KFold``

    Returns a tuple of ``(fold_ids, method_name)``.
    """
    # 1) Use predefined column if valid
    if "Fold" in df.columns:
        fold_ids = df["Fold"].values.astype(int)
        if _check_valid(fold_ids, y, n_splits):
            return fold_ids, "predefined"

    # 2) StratifiedGroupKFold via helper (up to 3 attempts)
    for attempt in range(3):
        seed = GLOBAL_RANDOM_SEED + attempt
        try:
            df_folds = group_stratified_kfold(
                df, n_splits=n_splits, random_state=seed, group_col=group_col
            )
            fold_ids = df_folds["Fold"].values.astype(int)
            if _check_valid(fold_ids, y, n_splits):
                return fold_ids, "stratified_group"
        except Exception:
            continue

    # 3) Plain GroupKFold (up to 3 attempts)
    for _ in range(3):
        gkf = GroupKFold(n_splits=n_splits)
        fold_ids = np.empty(len(df), dtype=int)
        for fold, (_, val_idx) in enumerate(gkf.split(df, groups=df[group_col])):
            fold_ids[val_idx] = fold
        if _check_valid(fold_ids, y, n_splits):
            return fold_ids, "group"

    # 4) Classic KFold
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=GLOBAL_RANDOM_SEED)
    fold_ids = np.empty(len(df), dtype=int)
    for fold, (_, val_idx) in enumerate(kf.split(df)):
        fold_ids[val_idx] = fold
    return fold_ids, "kfold"
