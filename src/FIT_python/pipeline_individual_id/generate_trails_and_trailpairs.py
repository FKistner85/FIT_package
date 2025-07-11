# Standard library
import random
from collections import defaultdict

# Third-party

import pandas as pd
from joblib import Parallel, delayed, load
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
from tqdm_joblib import tqdm_joblib
from itertools import combinations
from typing import List, Dict, Optional, Tuple, Union
import numpy as np
import pandas as pd
from FIT_python.config import PIPELINE_INDIVIDUAL_ID_CFG


def sample_trails(
    df: pd.DataFrame,
    all_ids: List[str],
    chunk_size: int,
    trail_size_list: List[int],
    max_trails_per_animal: Optional[int],
    rng: np.random.Generator,
) -> Dict[str, Dict[int, List[Tuple[int, List[int]]]]]:
    """Sample non-overlapping trails for each individual."""
    trails: Dict[str, Dict[int, List[Tuple[int, List[int]]]]] = {
        ind: {size: [] for size in trail_size_list} for ind in all_ids
    }
    for ind in all_ids:
        idxs = df.index[df["_group_id"] == ind].tolist()
        rng.shuffle(idxs)
        chunks = [idxs[i : i + chunk_size] for i in range(0, len(idxs), chunk_size)]
        for chunk_idx, chunk in enumerate(chunks):
            for size in trail_size_list:
                if len(chunk) >= size:
                    pool = trails[ind][size]
                    if max_trails_per_animal is None or len(pool) < max_trails_per_animal:
                        trail = rng.choice(chunk, size=size, replace=False).tolist()
                        pool.append((chunk_idx, trail))
    return trails


def build_pairwise_comparisons(
    trails_per_animal: Dict[str, Dict[int, List[Tuple[int, List[int]]]]],
    all_ids: List[str],
    sex_map: Dict[str, str],
    fallback_map: Dict[str, bool],
    trail_size_list: List[int],
    n_folds: int,
    random_state: int,
) -> Tuple[List[Dict], pd.DataFrame]:
    """Construct pairwise comparisons and summary statistics."""
    comparisons: List[Dict] = []

    # cross individual pairs
    for a, b in combinations(all_ids, 2):
        same_ind = "unknown" if fallback_map[a] or fallback_map[b] else False
        sex_a = sex_map.get(a, "unknown")
        sex_b = sex_map.get(b, "unknown")
        same_sex = "unknown" if fallback_map[a] or fallback_map[b] else (sex_a == sex_b)
        for size in trail_size_list:
            for chunk_i, ta in trails_per_animal[a][size]:
                for chunk_j, tb in trails_per_animal[b][size]:
                    comparisons.append(
                        {
                            "ind_a": a,
                            "ind_b": b,
                            "trail_a_id": f"{a}_{size}c{chunk_i}",
                            "trail_b_id": f"{b}_{size}c{chunk_j}",
                            "same_individual": same_ind,
                            "same_sex": same_sex,
                            "samples_a": ta,
                            "samples_b": tb,
                            "trail_size_a": size,
                            "trail_size_b": size,
                            "diff_size": 0,
                            "chunk_a": chunk_i,
                            "chunk_b": chunk_j,
                        }
                    )

    # within individual, cross chunk
    for ind in all_ids:
        same_ind = "unknown" if fallback_map[ind] else True
        same_sex = "unknown" if fallback_map[ind] else True
        all_trails = [
            (size, cidx, trail)
            for size in trail_size_list
            for cidx, trail in trails_per_animal[ind][size]
        ]
        for (size_a, ci, ta), (size_b, cj, tb) in combinations(all_trails, 2):
            if ci == cj:
                continue
            comparisons.append(
                {
                    "ind_a": ind,
                    "ind_b": ind,
                    "trail_a_id": f"{ind}_{size_a}c{ci}",
                    "trail_b_id": f"{ind}_{size_b}c{cj}",
                    "same_individual": same_ind,
                    "same_sex": same_sex,
                    "samples_a": ta,
                    "samples_b": tb,
                    "trail_size_a": size_a,
                    "trail_size_b": size_b,
                    "diff_size": abs(size_a - size_b),
                    "chunk_a": ci,
                    "chunk_b": cj,
                }
            )

    labels = [c["trail_size_a"] for c in comparisons]
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    fold_map: Dict[int, int] = {}
    for fold, (_, val_idx) in enumerate(skf.split(comparisons, labels)):
        for vi in val_idx:
            fold_map[vi] = fold
    for idx, comp in enumerate(comparisons):
        comp["fold"] = fold_map[idx]

    comp_df = pd.DataFrame(comparisons)
    summary_rows = []
    for size in trail_size_list:
        mask = comp_df["trail_size_a"] == size
        animals = pd.unique(comp_df.loc[mask, ["ind_a", "ind_b"]].values.ravel())
        n_anim = len(animals)
        n_trails = sum(len(trails_per_animal[ind][size]) for ind in animals)
        summary_rows.append({"sub_size": size, "n_animals": n_anim, "n_trails": n_trails})
    summary_rows.append(
        {"sub_size": "Total", "n_animals": len(all_ids), "n_trails": len(comp_df)}
    )
    summary_df = pd.DataFrame(summary_rows)

    same_counts = []
    diff_counts = []
    for ind in all_ids:
        same = comp_df[(comp_df.ind_a == ind) & (comp_df.ind_b == ind)].shape[0]
        diff = comp_df[
            ((comp_df.ind_a == ind) & (comp_df.ind_b != ind))
            | ((comp_df.ind_b == ind) & (comp_df.ind_a != ind))
        ].shape[0]
        same_counts.append(same)
        diff_counts.append(diff)
    avg_same = float(np.mean(same_counts))
    sd_same = float(np.std(same_counts, ddof=1)) if len(same_counts) > 1 else 0.0
    avg_diff = float(np.mean(diff_counts))
    sd_diff = float(np.std(diff_counts, ddof=1)) if len(diff_counts) > 1 else 0.0
    ratio = float(
        comp_df.same_individual.eq(True).sum()
        / max(1, comp_df.same_individual.eq(False).sum())
    )

    summary_df.loc[summary_df.sub_size == "Total", [
        "avg_comp_per_ind_same",
        "sd_comp_per_ind_same",
        "avg_comp_per_ind_diff",
        "sd_comp_per_ind_diff",
        "ratio_same_to_diff",
    ]] = [avg_same, sd_same, avg_diff, sd_diff, ratio]

    return comparisons, summary_df




def generate_pairwise_comparisons_from_df(
    df: pd.DataFrame,
    id_col: str = "individual_id",
    chunk_size: int = PIPELINE_INDIVIDUAL_ID_CFG.get("chunk_size", 7),
    trail_size_list: List[int] = PIPELINE_INDIVIDUAL_ID_CFG.get("trail_size_list", [7, 5, 3]),
    max_individuals: Optional[int] = PIPELINE_INDIVIDUAL_ID_CFG.get("max_individuals"),
    max_trails_per_animal: Optional[int] = PIPELINE_INDIVIDUAL_ID_CFG.get("max_trails_per_animal"),
    n_folds: int = PIPELINE_INDIVIDUAL_ID_CFG.get("n_folds", 3),
    random_state: int = PIPELINE_INDIVIDUAL_ID_CFG.get("random_state", 0),
    fallback_col: str = PIPELINE_INDIVIDUAL_ID_CFG.get("fallback_col", "trail"),
    sampling_mode: str = PIPELINE_INDIVIDUAL_ID_CFG.get("sampling_mode", "window"),
) -> Tuple[List[Dict], pd.DataFrame]:
    """Generate all pairwise trail comparisons.

    Steps
    -----
    1. Build ``group_id`` equal to ``individual_id`` or ``fallback_col`` if the former is missing.
    2. For each ``group_id`` sample non-overlapping chunks of length ``chunk_size``
       and create trails of lengths given in ``trail_size_list``.
    3. Construct all cross-individual pairs and all within-individual cross-chunk pairs.
    4. ``same_individual`` is ``True``/``False`` or ``"unknown"`` when a fallback id is used.
    5. ``same_sex`` behaves analogously.
    6. Perform ``StratifiedKFold`` by ``trail_size_a``.
    7. Produce a summary table including average and standard deviation of pair counts.

    ``sampling_mode='predefined'`` automatically interprets ``fallback_col`` as pre-built trails.
    """
    rng = np.random.default_rng(random_state)
    df = df.copy()

    # --- 0) prepare group_id, sex_map and fallback_map ---
    orig = df[id_col]
    if fallback_col not in df.columns:
        raise ValueError(f"fallback column '{fallback_col}' not found in DataFrame")
    # group_id: original id if available and not "unknown", otherwise ``fallback_col``
    df['_group_id'] = orig.where(orig.notna() & (orig != "unknown"),
                                 df[fallback_col]).astype(str)
    # sex_map: first non-null sex per group_id, otherwise "unknown"
    sex_map = (
        df
        .set_index('_group_id')['sex']
        .fillna("unknown")
        .replace("", "unknown")
        .to_dict()
    )
    # fallback_map: True for all group_ids originating from ``fallback_col``
    fallback_gids = df.loc[orig.isna() | (orig == "unknown"), '_group_id'].unique().tolist()
    fallback_map = {gid: True for gid in fallback_gids}
    for gid in df['_group_id'].unique():
        fallback_map.setdefault(gid, False)

    # --- 1) restrict animals if requested ---
    all_ids = df['_group_id'].unique().tolist()
    if max_individuals is not None and len(all_ids) > max_individuals:
        all_ids = rng.choice(all_ids, size=max_individuals, replace=False).tolist()

    if sampling_mode not in {"predefined", "window"}:
        raise ValueError("sampling_mode must be 'predefined' or 'window'")

    if sampling_mode == "predefined":
        trails_per_animal = {}
        for gid, grp in df[df['_group_id'].isin(all_ids)].groupby("_group_id"):
            by_trail = {}
            for trail_name, sub in grp.groupby(fallback_col):
                idxs = sub.index.tolist()
                if idxs:
                    size = len(idxs)
                    by_trail.setdefault(size, []).append((0, idxs))
            trails_per_animal[str(gid)] = by_trail
    else:
        trails_per_animal = sample_trails(
            df,
            all_ids,
            chunk_size,
            trail_size_list,
            max_trails_per_animal,
            rng,
        )

    comparisons, summary_df = build_pairwise_comparisons(
        trails_per_animal,
        all_ids,
        sex_map,
        fallback_map,
        trail_size_list,
        n_folds,
        random_state,
    )

    return comparisons, summary_df