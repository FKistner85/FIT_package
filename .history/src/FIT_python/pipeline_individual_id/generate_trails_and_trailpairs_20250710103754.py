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




from itertools import combinations
from typing import List, Dict, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

def generate_pairwise_comparisons_from_df(
    df: pd.DataFrame,
    id_col: str = "individual_id",
    chunk_size: int = 7,
    trail_size_list: Optional[List[int]] = None,
    max_individuals: Optional[int] = None,
    max_trails_per_animal: Optional[int] = None,
    n_folds: int = 3,
    random_state: int = 0,
    fallback_col: str = "trail",
    use_default_trails: bool = False,
    n_samples_per_trail: int = 1
) -> Tuple[List[Dict], pd.DataFrame]:
    """
    Generate all pairwise trail comparisons.

    Neuerungen:
      * trail_size_list=None erlaubt auto-Ermittlung der Größen aus Default-Trails.
      * n_samples_per_trail steuert, wie oft pro Trail/Sub-Trail gesampelt wird.
    """
    rng = np.random.default_rng(random_state)
    df = df.copy()

    # --- 0) group_id & Maps ---
    if fallback_col not in df.columns:
        raise ValueError(f"fallback column '{fallback_col}' not found")
    df["_group_id"] = (
        df[id_col]
        .where(df[id_col].notna() & (df[id_col] != "unknown"), df[fallback_col])
        .astype(str)
    )
    sex_map = (
        df.set_index("_group_id")["sex"]
          .fillna("unknown")
          .replace("", "unknown")
          .to_dict()
    )
    fallback_gids = df.loc[
        df[id_col].isna() | (df[id_col] == "unknown"), "_group_id"
    ].unique().tolist()
    fallback_map = {gid: True for gid in fallback_gids}
    for gid in df["_group_id"].unique():
        fallback_map.setdefault(gid, False)

    # --- 1) limit individuals ---
    all_ids = df["_group_id"].unique().tolist()
    if max_individuals and len(all_ids) > max_individuals:
        all_ids = rng.choice(all_ids, size=max_individuals, replace=False).tolist()

    # --- 2) determine trail sizes when using default ---
    if use_default_trails:
        # finde alle tatsächlichen Trail-Indizes pro group
        trail_indices = {
            (ind, trail_name): grp.index.to_list()
            for ind, grp in df.groupby(["tr", fallback_col])
        }
        # autosize, falls None
        if trail_size_list is None:
            trail_size_list = sorted({len(idxs) for idxs in trail_indices.values()})
    else:
        if trail_size_list is None:
            raise ValueError("trail_size_list darf nur None sein, wenn use_default_trails=True")

    # --- 3) Build trails_per_animal ---
    trails_per_animal: Dict[str, Dict[int, List[Tuple[int, List[int]]]]] = {
        ind: {size: [] for size in trail_size_list} for ind in all_ids
    }

    if use_default_trails:
        for (ind, trail_name), idxs in trail_indices.items():
            L = len(idxs)
            for size in trail_size_list:
                if size <= L:
                    pool = trails_per_animal[ind][size]
                    # wie oft sampeln?
                    for rep in range(n_samples_per_trail):
                        if max_trails_per_animal is None or len(pool) < max_trails_per_animal:
                            # n_samples_per_trail-Sub-Sample
                            sub = rng.choice(idxs, size=size, replace=False).tolist()
                            # chunk_idx hier = rep
                            pool.append((rep, sub))
    else:
        for ind in all_ids:
            idxs = df.index[df["_group_id"] == ind].tolist()
            rng.shuffle(idxs)
            chunks = [idxs[i : i + chunk_size] for i in range(0, len(idxs), chunk_size)]
            for chunk_idx, chunk in enumerate(chunks):
                for size in trail_size_list:
                    if len(chunk) >= size:
                        pool = trails_per_animal[ind][size]
                        for rep in range(n_samples_per_trail):
                            if max_trails_per_animal is None or len(pool) < max_trails_per_animal:
                                sub = rng.choice(chunk, size=size, replace=False).tolist()
                                pool.append((chunk_idx, sub))

    # --- 4) Cross-Individual-Paare ---
    comparisons: List[Dict] = []
    for a, b in combinations(all_ids, 2):
        same_ind = "unknown" if fallback_map[a] or fallback_map[b] else False
        same_sex = (
            "unknown" if fallback_map[a] or fallback_map[b]
            else sex_map[a] == sex_map[b]
        )
        for size in trail_size_list:
            for ci, ta in trails_per_animal[a][size]:
                for cj, tb in trails_per_animal[b][size]:
                    comparisons.append({
                        "ind_a":           a,
                        "ind_b":           b,
                        "trail_a_id":      f"{a}_{size}c{ci}",
                        "trail_b_id":      f"{b}_{size}c{cj}",
                        "same_individual": same_ind,
                        "same_sex":        same_sex,
                        "samples_a":       ta,
                        "samples_b":       tb,
                        "trail_size_a":    size,
                        "trail_size_b":    size,
                        "diff_size":       0,
                        "chunk_a":         ci,
                        "chunk_b":         cj,
                    })

    # --- 5) Within-Individual cross-chunk ---
    for ind in all_ids:
        same_ind = not fallback_map[ind]
        same_sex = same_ind
        all_trails = [
            (size, ci, tr)
            for size in trail_size_list
            for ci, tr in trails_per_animal[ind][size]
        ]
        for (sa, cia, ta), (sb, cib, tb) in combinations(all_trails, 2):
            if cia == cib:
                continue
            comparisons.append({
                "ind_a":           ind,
                "ind_b":           ind,
                "trail_a_id":      f"{ind}_{sa}c{cia}",
                "trail_b_id":      f"{ind}_{sb}c{cib}",
                "same_individual": same_ind,
                "same_sex":        same_sex,
                "samples_a":       ta,
                "samples_b":       tb,
                "trail_size_a":    sa,
                "trail_size_b":    sb,
                "diff_size":       abs(sa - sb),
                "chunk_a":         cia,
                "chunk_b":         cib,
            })

    # --- 6) StratifiedKFold by trail_size_a ---
    labels = [c["trail_size_a"] for c in comparisons]
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    fold_map: Dict[int, int] = {}
    for fold, (_, val_idx) in enumerate(skf.split(comparisons, labels)):
        for vi in val_idx:
            fold_map[vi] = fold
    for idx, comp in enumerate(comparisons):
        comp["fold"] = fold_map[idx]

    # --- 7) Summary ---
    comp_df = pd.DataFrame(comparisons)
    summary_rows = []
    for size in trail_size_list:
        mask = comp_df["trail_size_a"] == size
        animals = pd.unique(comp_df.loc[mask, ["ind_a","ind_b"]].values.ravel())
        summary_rows.append({
            "sub_size":  size,
            "n_animals": len(animals),
            "n_trails":  sum(len(trails_per_animal[ind][size]) for ind in animals)
        })
    summary_rows.append({"sub_size": "Total", "n_animals": len(all_ids), "n_trails": len(comp_df)})
    summary_df = pd.DataFrame(summary_rows)

    # Avg/SD & Ratio
    same_counts, diff_counts = [], []
    for ind in all_ids:
        same = comp_df[(comp_df.ind_a==ind)&(comp_df.ind_b==ind)].shape[0]
        diff = comp_df[
            ((comp_df.ind_a==ind)&(comp_df.ind_b!=ind))|
            ((comp_df.ind_b==ind)&(comp_df.ind_a!=ind))
        ].shape[0]
        same_counts.append(same)
        diff_counts.append(diff)
    avg_same = float(np.mean(same_counts))
    sd_same  = float(np.std(same_counts, ddof=1)) if len(same_counts)>1 else 0.0
    avg_diff = float(np.mean(diff_counts))
    sd_diff  = float(np.std(diff_counts, ddof=1)) if len(diff_counts)>1 else 0.0
    ratio    = float(comp_df.same_individual.eq(True).sum() /
                      max(1, comp_df.same_individual.eq(False).sum()))

    summary_df.loc[summary_df.sub_size=="Total", [
        "avg_comp_per_ind_same","sd_comp_per_ind_same",
        "avg_comp_per_ind_diff","sd_comp_per_ind_diff","ratio_same_to_diff"
    ]] = [avg_same, sd_same, avg_diff, sd_diff, ratio]

    return comparisons, summary_df
