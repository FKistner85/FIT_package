"""Utilities for data splitting and cross-validation."""
from typing import Iterable, List, Dict
import numpy as np
from sklearn.model_selection import StratifiedKFold

__all__ = ["sequential_holdout_ids", "assign_folds"]


def sequential_holdout_ids(
    unique_ids: Iterable[str],
    val_sizes: Iterable[int] = (2, 4, 6, 8),
    n_iter: int = 1,
    random_state: int | None = None,
) -> List[Dict[str, List[str]]]:
    """Generate sequential train/validation splits based on unique IDs."""
    ids = list(unique_ids)
    if len(ids) < 3:
        raise ValueError("Need at least three unique IDs for holdouts")

    rng = np.random.default_rng(random_state)
    results = []
    for it in range(n_iter):
        for n_val in val_sizes:
            n_val_eff = max(2, min(n_val, len(ids) - 1))
            val_ids = list(rng.choice(ids, size=n_val_eff, replace=False))
            train_ids = [i for i in ids if i not in val_ids]
            results.append(
                {
                    "iteration": it,
                    "n_val": n_val_eff,
                    "val_ids": val_ids,
                    "train_ids": train_ids,
                }
            )
    return results


def assign_folds(
    comparisons: Iterable[Dict],
    sex_map: Dict[str, str],
    *,
    n_folds: int = 5,
    random_state: int = 0,
) -> List[Dict]:
    """Assign fold numbers to comparisons based on individual IDs."""
    ids = sorted(sex_map)
    id_labels = [sex_map[i] for i in ids]
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    id_to_fold: Dict[str, int] = {}
    for fold, (_, val_idx) in enumerate(skf.split(ids, id_labels)):
        for vi in val_idx:
            id_to_fold[ids[vi]] = fold

    filtered = []
    for comp in comparisons:
        fa = id_to_fold.get(comp["ind_a"])
        fb = id_to_fold.get(comp["ind_b"])
        if fa is None or fb is None or fa != fb:
            continue
        c2 = dict(comp)
        c2["fold"] = fa
        filtered.append(c2)
    return filtered
