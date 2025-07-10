from itertools import combinations
from typing import Dict, List, Optional, Tuple, Literal

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold


def sample_trails(
    df: pd.DataFrame,
    group_col: str,
    mode: Literal["predefined", "window"],
    *,
    trails_per_animal: Optional[Dict] = None,
    window_lengths: Optional[List[int]] = None,
    n_windows_per_length: int = 5,
    random_state: int = 0,
) -> Dict[str, Dict[int, List[Tuple[int, List[int]]]]]:
    """Return ``trails_per_animal`` for downstream pairing.

    ``predefined`` simply returns the provided ``trails_per_animal`` structure.
    ``window`` generates all contiguous windows of ``window_lengths`` per
    ``group_col`` and randomly selects ``n_windows_per_length`` windows.
    """

    rng = np.random.default_rng(random_state)

    if mode == "predefined":
        if trails_per_animal is None:
            raise ValueError("trails_per_animal must be provided for mode='predefined'")
        return trails_per_animal

    if mode != "window":
        raise ValueError("mode must be 'predefined' or 'window'")

    if window_lengths is None:
        raise ValueError("window_lengths must be provided for mode='window'")

    pools: Dict[str, Dict[int, List[Tuple[int, List[int]]]]] = {}
    for gid, grp in df.groupby(group_col):
        idxs = sorted(grp.index.tolist())
        gid = str(gid)
        pools[gid] = {L: [] for L in window_lengths}
        for L in window_lengths:
            windows = [idxs[i : i + L] for i in range(0, len(idxs) - L + 1)]
            if not windows:
                continue
            choices = rng.choice(
                len(windows),
                size=min(n_windows_per_length, len(windows)),
                replace=False,
            )
            for ci in np.asarray(choices, dtype=int):
                pools[gid][L].append((int(ci), windows[ci]))

    return pools


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
    *,
    sampling_mode: Literal["predefined", "window"] = "predefined",
    trails_per_animal: Optional[Dict] = None,
    window_lengths: Optional[List[int]] = None,
    n_windows_per_length: int = 5,
) -> Tuple[List[Dict], pd.DataFrame]:
    """Generate trail pairs with metadata.

    1) Erzeuge ``group_id`` aus ``id_col`` bzw. ``fallback_col``.
    2) Erzeuge ``trails_per_animal`` entweder aus vorgegebenen Pools oder per
       Sliding-Window-Sampling.
    3) Baue alle Cross- und Within-Individual-Paare.
    4) ``same_individual`` und ``same_sex`` markieren Gleichheit oder
       "unknown".
    5) ``StratifiedKFold`` nach ``trail_size_a``.
    6) Summary-Tabelle mit pro-Länge und Total-Zeile inkl. avg/sd Pair counts.
    """
    df = df.copy()

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


    if sampling_mode == "predefined":
        trails_per_animal = sample_trails(
            df,
            group_col="_group_id",
            mode="predefined",
            trails_per_animal=trails_per_animal,
        )
    else:
        trails_per_animal = sample_trails(
            df,
            group_col="_group_id",
            mode="window",
            window_lengths=window_lengths,
            n_windows_per_length=n_windows_per_length,
            random_state=random_state,
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
