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


from itertools import combinations
from typing import List, Dict, Optional, Tuple, Union
import numpy as np
import pandas as pd




def generate_pairwise_comparisons_from_df(
    df: pd.DataFrame,
    id_col: str = "individual_id",
    chunk_size: int = 7,
    trail_size_list: List[int] = [7, 5, 3],
    max_individuals: Optional[int] = None,
    max_trails_per_animal: Optional[int] = None,
    n_folds: int = 5,
    random_state: int = 0,
    fallback_col: str = "trail"
) -> Tuple[List[Dict], pd.DataFrame]:
    """
    1) Erzeuge group_id = individual_id, bzw. wenn NaN/unknown dann fallback trail.
    2) Sample pro group_id nicht-überlappende Chunks der Länge chunk_size,
       daraus bis zu max_trails_per_animal Trails jeder Länge in trail_size_list.
    3) Baue alle Cross-Individual-Paare UND alle Within-Individual, cross-chunk Paare.
    4) same_individual = True/False, oder "unknown" wenn eine Seite fallback benutzt.
    5) same_sex = True/False/"unknown" analog.
    6) StratifiedKFold nach trail_size_a.
    7) Summary-Tabelle mit pro-Länge und Total-Zeile inkl. avg/sd Pair counts.
    """
    rng = np.random.default_rng(random_state)
    df = df.copy()

    # --- 0) group_id und sex_map, fallback_map ---
    orig = df[id_col]
    if fallback_col not in df.columns:
        raise ValueError(f"fallback column '{fallback_col}' not found in DataFrame")
    # group_id: original id wenn vorhanden & != "unknown", sonst trail
    df['_group_id'] = orig.where(orig.notna() & (orig != "unknown"),
                                 df[fallback_col]).astype(str)
    # sex_map: erster non-null sex pro group_id, unknown sonst
    sex_map = (
        df
        .set_index('_group_id')['sex']
        .fillna("unknown")
        .replace("", "unknown")
        .to_dict()
    )
    # fallback_map: True für alle group_ids, die aus fallback_col kamen
    fallback_gids = df.loc[orig.isna() | (orig == "unknown"), '_group_id'].unique().tolist()
    fallback_map = {gid: True for gid in fallback_gids}
    for gid in df['_group_id'].unique():
        fallback_map.setdefault(gid, False)

    # --- 1) Tiere einschränken ---
    all_ids = df['_group_id'].unique().tolist()
    if max_individuals is not None and len(all_ids) > max_individuals:
        all_ids = rng.choice(all_ids, size=max_individuals, replace=False).tolist()

    # --- 2) Chunks & Trails pro group_id ---
    trails_per_animal: Dict[str, Dict[int, List[Tuple[int,List[int]]]]] = {
        ind: {size: [] for size in trail_size_list}
        for ind in all_ids
    }
    for ind in all_ids:
        idxs = df.index[df['_group_id'] == ind].tolist()
        rng.shuffle(idxs)
        # in non-overlapping Chunks
        chunks = [idxs[i:i+chunk_size] for i in range(0, len(idxs), chunk_size)]
        for chunk_idx, chunk in enumerate(chunks):
            for size in trail_size_list:
                if len(chunk) >= size:
                    pool = trails_per_animal[ind][size]
                    if max_trails_per_animal is None or len(pool) < max_trails_per_animal:
                        trail = rng.choice(chunk, size=size, replace=False).tolist()
                        pool.append((chunk_idx, trail))

    # --- 3) Cross-Individual-Paare ---
    comparisons: List[Dict] = []
    for a, b in combinations(all_ids, 2):
        # same_individual unknown falls fallback, sonst False
        same_ind = ("unknown" if fallback_map[a] or fallback_map[b] else False)
        # same_sex analog
        sex_a = sex_map.get(a, "unknown")
        sex_b = sex_map.get(b, "unknown")
        if fallback_map[a] or fallback_map[b]:
            same_sex = "unknown"
        else:
            same_sex = (sex_a == sex_b)
        for size in trail_size_list:
            for chunk_i, ta in trails_per_animal[a][size]:
                for chunk_j, tb in trails_per_animal[b][size]:
                    comparisons.append({
                        "ind_a":           a,
                        "ind_b":           b,
                        "trail_a_id":      f"{a}_{size}c{chunk_i}",
                        "trail_b_id":      f"{b}_{size}c{chunk_j}",
                        "same_individual": same_ind,
                        "same_sex":        same_sex,
                        "samples_a":       ta,
                        "samples_b":       tb,
                        "trail_size_a":    size,
                        "trail_size_b":    size,
                        "diff_size":       abs(size - size),
                        "chunk_a":         chunk_i,
                        "chunk_b":         chunk_j,
                    })

    # --- 4) Within-Individual, cross-chunk ---
    for ind in all_ids:
        # same_individual = unknown bei fallback, sonst True
        same_ind = ("unknown" if fallback_map[ind] else True)
        sex_i = sex_map.get(ind, "unknown")
        same_sex = ("unknown" if fallback_map[ind] else True)
        # alle Trails unterschiedlicher chunk
        all_trails = [
            (size, cidx, trail)
            for size in trail_size_list
            for cidx, trail in trails_per_animal[ind][size]
        ]
        for (size_a, ci, ta), (size_b, cj, tb) in combinations(all_trails, 2):
            if ci == cj:
                continue
            comparisons.append({
                "ind_a":           ind,
                "ind_b":           ind,
                "trail_a_id":      f"{ind}_{size_a}c{ci}",
                "trail_b_id":      f"{ind}_{size_b}c{cj}",
                "same_individual": same_ind,
                "same_sex":        same_sex,
                "samples_a":       ta,
                "samples_b":       tb,
                "trail_size_a":    size_a,
                "trail_size_b":    size_b,
                "diff_size":       abs(size_a - size_b),
                "chunk_a":         ci,
                "chunk_b":         cj,
            })

    # --- 5) StratifiedKFold nach trail_size_a ---
    labels = [c["trail_size_a"] for c in comparisons]
    skf    = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    fold_map: Dict[int,int] = {}
    for fold, (_, val_idx) in enumerate(skf.split(comparisons, labels)):
        for vi in val_idx:
            fold_map[vi] = fold
    for idx, comp in enumerate(comparisons):
        comp["fold"] = fold_map[idx]

    # --- 6) Summary bauen ---
    comp_df = pd.DataFrame(comparisons)
    summary_rows = []
    for size in trail_size_list:
        mask = comp_df["trail_size_a"] == size
        animals = pd.unique(comp_df.loc[mask, ["ind_a","ind_b"]].values.ravel())
        n_anim   = len(animals)
        n_trails = sum(len(trails_per_animal[ind][size]) for ind in animals)
        summary_rows.append({
            "sub_size": size,
            "n_animals": n_anim,
            "n_trails": n_trails
        })
    # Total-Zeile
    summary_rows.append({
        "sub_size":  "Total",
        "n_animals": len(all_ids),
        "n_trails":  len(comp_df)
    })
    summary_df = pd.DataFrame(summary_rows)

    # --- 7) Ergänze avg/sd Vergleiche pro Individuum und ratio same/diff ---
    same_counts = []
    diff_counts = []
    for ind in all_ids:
        same = comp_df[(comp_df.ind_a==ind)&(comp_df.ind_b==ind)].shape[0]
        diff = comp_df[
            ((comp_df.ind_a==ind)&(comp_df.ind_b!=ind)) |
            ((comp_df.ind_b==ind)&(comp_df.ind_a!=ind))
        ].shape[0]
        same_counts.append(same)
        diff_counts.append(diff)
    avg_same = float(np.mean(same_counts))
    sd_same  = float(np.std(same_counts, ddof=1)) if len(same_counts)>1 else 0.0
    avg_diff = float(np.mean(diff_counts))
    sd_diff  = float(np.std(diff_counts, ddof=1)) if len(diff_counts)>1 else 0.0
    ratio    = float(comp_df.same_individual.eq(True).sum() / max(1, comp_df.same_individual.eq(False).sum()))

    # Schreibe in die Total-Zeile
    summary_df.loc[summary_df.sub_size=="Total", [
        "avg_comp_per_ind_same",
        "sd_comp_per_ind_same",
        "avg_comp_per_ind_diff",
        "sd_comp_per_ind_diff",
        "ratio_same_to_diff"
    ]] = [avg_same, sd_same, avg_diff, sd_diff, ratio]

    return comparisons, summary_df