from itertools import combinations
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold


def sample_trails_per_animal(
    df: pd.DataFrame,
    id_col: str,
    artificial_trail_size: int,
    trail_size_list: Optional[List[int]],
    num_individuals: Optional[int],
    strict_individuals: bool,
    max_trails_per_animal: Optional[int],
    n_samples_per_trail: int,
    random_state: int,
    fallback_col: str,
    use_default_trails: bool,
) -> Tuple[Dict[str, Dict[int, List[Tuple[int, List[int]]]]], List[str]]:
    """Sample sub-trails for every individual."""

    rng = np.random.default_rng(random_state)

    if fallback_col not in df.columns:
        raise ValueError(f"fallback column '{fallback_col}' not found")

    df = df.copy()
    df["_group_id"] = (
        df[id_col]
        .where(df[id_col].notna() & (df[id_col] != "unknown"), df[fallback_col])
        .astype(str)
    )

    all_ids = list(df["_group_id"].unique())
    if num_individuals is not None:
        if strict_individuals and len(all_ids) < num_individuals:
            raise ValueError(
                f"{len(all_ids)} Individuen vorhanden, aber num_individuals={num_individuals} verlangt."
            )
        n_pick = min(len(all_ids), num_individuals)
        if n_pick < len(all_ids):
            all_ids = rng.choice(all_ids, size=n_pick, replace=False).tolist()

    if use_default_trails:
        trail_indices = {
            (ind, tname): grp.index.tolist()
            for (ind, tname), grp in df.groupby(["_group_id", fallback_col])
            if ind in all_ids
        }
        if trail_size_list is None:
            trail_size_list = sorted({len(idxs) for idxs in trail_indices.values()})
    else:
        if trail_size_list is None:
            raise ValueError(
                "trail_size_list darf nur None sein, wenn use_default_trails=True"
            )

    trails_per_animal: Dict[str, Dict[int, List[Tuple[int, List[int]]]]] = {
        ind: {size: [] for size in trail_size_list} for ind in all_ids
    }

    if use_default_trails:
        for (ind, _), idxs in trail_indices.items():
            L = len(idxs)
            for size in trail_size_list:
                if size <= L:
                    pool = trails_per_animal[ind][size]
                    for rep in range(n_samples_per_trail):
                        if max_trails_per_animal is None or len(pool) < max_trails_per_animal:
                            sub = rng.choice(idxs, size=size, replace=False).tolist()
                            pool.append((rep, sub))
    else:
        for ind in all_ids:
            idxs = df.index[df["_group_id"] == ind].tolist()
            rng.shuffle(idxs)
            chunks = [idxs[i:i + artificial_trail_size] for i in range(0, len(idxs), artificial_trail_size)]
            for chunk_idx, chunk in enumerate(chunks):
                for size in trail_size_list:
                    if len(chunk) >= size:
                        pool = trails_per_animal[ind][size]
                        for rep in range(n_samples_per_trail):
                            if max_trails_per_animal is None or len(pool) < max_trails_per_animal:
                                sub = rng.choice(chunk, size=size, replace=False).tolist()
                                pool.append((chunk_idx, sub))

    return trails_per_animal, all_ids


def build_pairwise_comparisons(
    trails_per_animal: Dict[str, Dict[int, List[Tuple[int, List[int]]]]],
    df: pd.DataFrame,
    id_col: str,
    sex_map: Dict[str, str],
    fallback_map: Dict[str, bool],
    n_folds: int,
    random_state: int,
) -> Tuple[List[Dict], pd.DataFrame]:
    """Create cross- and within-individual comparisons and summary."""

    trail_size_list = sorted(next(iter(trails_per_animal.values())).keys()) if trails_per_animal else []
    all_ids = list(trails_per_animal.keys())

    comparisons: List[Dict] = []

    for a, b in combinations(all_ids, 2):
        ids_a = df.loc[df["_group_id"] == a, id_col].dropna().unique()
        ids_b = df.loc[df["_group_id"] == b, id_col].dropna().unique()

        if not ids_a.size or ids_a[0] == "unknown" or not ids_b.size or ids_b[0] == "unknown":
            same_ind = "unknown"
        else:
            same_ind = str(ids_a[0] == ids_b[0])

        if fallback_map[a] or fallback_map[b]:
            same_sex = "unknown"
        else:
            same_sex = str(sex_map[a] == sex_map[b])

        for size in trail_size_list:
            for ci, ta in trails_per_animal[a][size]:
                for cj, tb in trails_per_animal[b][size]:
                    comparisons.append(
                        {
                            "ind_a": a,
                            "ind_b": b,
                            "trail_a_id": f"{a}_{size}c{ci}",
                            "trail_b_id": f"{b}_{size}c{cj}",
                            "same_individual": same_ind,
                            "same_sex": same_sex,
                            "samples_a": ta,
                            "samples_b": tb,
                            "trail_size_a": size,
                            "trail_size_b": size,
                            "diff_size": 0,
                            "chunk_a": ci,
                            "chunk_b": cj,
                        }
                    )

    for ind in all_ids:
        same_ind = True
        same_sex = True

        all_trails = [
            (size, ci, tr)
            for size in trail_size_list
            for ci, tr in trails_per_animal[ind][size]
        ]

        for (sa, cia, ta), (sb, cib, tb) in combinations(all_trails, 2):
            if cia == cib:
                continue
            comparisons.append(
                {
                    "ind_a": ind,
                    "ind_b": ind,
                    "trail_a_id": f"{ind}_{sa}c{cia}",
                    "trail_b_id": f"{ind}_{sb}c{cib}",
                    "same_individual": same_ind,
                    "same_sex": same_sex,
                    "samples_a": ta,
                    "samples_b": tb,
                    "trail_size_a": sa,
                    "trail_size_b": sb,
                    "diff_size": abs(sa - sb),
                    "chunk_a": cia,
                    "chunk_b": cib,
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
        summary_rows.append(
            {
                "sub_size": size,
                "n_animals": len(animals),
                "n_trails": sum(len(trails_per_animal[ind][size]) for ind in animals),
            }
        )
    summary_rows.append({"sub_size": "Total", "n_animals": len(all_ids), "n_trails": len(comp_df)})
    summary_df = pd.DataFrame(summary_rows)

    same_counts, diff_counts = [], []
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
    ratio = float(comp_df.same_individual.eq(True).sum() / max(1, comp_df.same_individual.eq(False).sum()))

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
    artificial_trail_size: int = 10,
    trail_size_list: Optional[List[int]] = None,
    num_individuals: Optional[int] = None,
    strict_individuals: bool = False,
    max_trails_per_animal: Optional[int] = None,
    n_samples_per_trail: int = 1,
    n_folds: int = 3,
    random_state: int = 0,
    fallback_col: str = "trail",
    use_default_trails: bool = False,
    simple_pairs_col: Optional[str] = None,
) -> Tuple[List[Dict], pd.DataFrame]:
    """Generate trail pairs with metadata.

    1) Erzeuge group_id = individual_id, bzw. wenn NaN/unknown dann fallback trail.
    2) Sample pro group_id nicht-überlappende Chunks der Länge chunk_size,
       daraus bis zu max_trails_per_animal Trails jeder Länge in trail_size_list.
    3) Baue alle Cross-Individual-Paare UND alle Within-Individual, cross-chunk Paare.
    4) same_individual = True/False, oder "unknown" wenn eine Seite fallback benutzt.
    5) same_sex = True/False/"unknown" analog.
    6) StratifiedKFold nach trail_size_a.
    7) Summary-Tabelle mit pro-Länge und Total-Zeile inkl. avg/sd Pair counts.
    
    Ist ``simple_pairs_col`` gesetzt, wird stattdessen eine vereinfachte Liste
    aller Paarungen dieser Spalte zurückgegeben und die oben beschriebenen
    Schritte werden übersprungen.
    """
    rng = np.random.default_rng(random_state)
    df = df.copy()

    # -- simple pair generation without sampling --
    if simple_pairs_col is not None:
        if simple_pairs_col not in df.columns:
            raise ValueError(f"column '{simple_pairs_col}' not found")
        values = df[simple_pairs_col].dropna().astype(str).unique().tolist()
        comps = [
            {"ind_a": a, "ind_b": b}
            for a in values
            for b in values
            if a != b
        ]
        summary_df = pd.DataFrame({"n_pairs": [len(comps)]})
        return comps, summary_df

    # --- 0) group_id + sex_map + fallback_map ---
    if fallback_col not in df.columns:
        raise ValueError(f"fallback column '{fallback_col}' not found")
    df["_group_id"] = (
        df[id_col]
        .where(df[id_col].notna() & (df[id_col] != "unknown"), df[fallback_col])
        .astype(str)
    )
    # sex_map: per group_id
    sex_map = (
        df.set_index("_group_id")["sex"]
          .fillna("unknown")
          .replace("", "unknown")
          .to_dict()
    )
    # fallback_map: True, wenn original id fehlte
    fallback_map = {
        gid: (orig is pd.NA or orig == "unknown")
        for gid, orig in zip(df["_group_id"], df[id_col])
    }


    trails_per_animal, all_ids = sample_trails_per_animal(
        df,
        id_col,
        artificial_trail_size,
        trail_size_list,
        num_individuals,
        strict_individuals,
        max_trails_per_animal,
        n_samples_per_trail,
        random_state,
        fallback_col,
        use_default_trails,
    )

    comparisons, summary_df = build_pairwise_comparisons(
        trails_per_animal,
        df,
        id_col,
        sex_map,
        fallback_map,
        n_folds,
        random_state,
    )

    return comparisons, summary_df
