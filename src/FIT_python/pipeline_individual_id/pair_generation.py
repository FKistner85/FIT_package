from itertools import combinations
from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np

__all__ = ["build_pairwise_comparisons"]


def build_pairwise_comparisons(
    df: pd.DataFrame,
    id_col: str,
    sex_map: Dict[str, str],
    fallback_map: Dict[str, bool],
    *,
    trails_per_animal: Optional[
        Dict[str, Dict[int, List[Tuple[int, List[str]]]]]
    ] = None,
    trail_col: Optional[str] = None,
) -> Tuple[List[Dict], pd.DataFrame]:
    """Create cross- and within-individual comparisons without folds.

    Either ``trails_per_animal`` must be supplied or ``trail_col`` must be the
    name of a column in ``df`` that already groups observations into trails.
    """
    if trails_per_animal is None:
        if trail_col is None:
            raise ValueError("Either trails_per_animal or trail_col must be provided")

        trails_per_animal = {}
        for (gid, tid), grp in df.groupby(["_group_id", trail_col]):
            size = len(grp)
            trails_per_animal.setdefault(str(gid), {}).setdefault(size, []).append(
                (str(tid), grp[id_col].astype(str).tolist())
            )

    trail_size_list = (
        sorted(next(iter(trails_per_animal.values())).keys())
        if trails_per_animal
        else []
    )
    all_ids = list(trails_per_animal.keys())

    comparisons: List[Dict] = []

    for a, b in combinations(all_ids, 2):
        ids_a = df.loc[df["_group_id"] == a, id_col].dropna().unique()
        ids_b = df.loc[df["_group_id"] == b, id_col].dropna().unique()

        if (
            not ids_a.size
            or ids_a[0] == "unknown"
            or not ids_b.size
            or ids_b[0] == "unknown"
        ):
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
        same_ind = "True"
        same_sex = "True"

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
    summary_rows.append(
        {"sub_size": "Total", "n_animals": len(all_ids), "n_trails": len(comp_df)}
    )
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
    ratio = float(
        (comp_df.same_individual == "True").sum()
        / max(1, (comp_df.same_individual == "False").sum())
    )

    summary_df.loc[
        summary_df.sub_size == "Total",
        [
            "avg_comp_per_ind_same",
            "sd_comp_per_ind_same",
            "avg_comp_per_ind_diff",
            "sd_comp_per_ind_diff",
            "ratio_same_to_diff",
        ],
    ] = [avg_same, sd_same, avg_diff, sd_diff, ratio]

    return comparisons, summary_df
