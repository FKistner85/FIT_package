from itertools import combinations
from typing import Dict, List, Optional, Tuple, Literal
import math

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from .geometric_pairwise_projection import (
    generate_pairwise_comparisons_from_df as _generate_pw_from_df,
)

__all__ = [
    "sample_trails",
    "build_pairwise_comparisons",
    "generate_pairwise_comparisons_from_df",
]


def sample_trails(
    df: pd.DataFrame,
    group_col: str,
    window_lengths: List[int],
    N_pool: int,
    n_windows: int,
    random_state: int,
    *,
    id_col: str = "id",
) -> Dict[str, Dict[int, List[List[str]]]]:
    """Sample diverse ``id`` subsets for each individual.

    Parameters
    ----------
    df : pd.DataFrame
        Input data containing one row per footprint.
    group_col : str
        Column to group by (usually the prepared ``_group_id``).
    window_lengths : list of int
        Desired trail lengths.
    N_pool : int
        Number of random subsets to draw **without replacement** per
        individual and length.
    n_windows : int
        Number of final subsets to return per individual and length.
    random_state : int
        Seed for the random number generator.
    id_col : str, optional
        Column containing unique IDs for each footprint. Defaults to ``"id"``.

    The function draws a pool of subsets for every ``group_col`` and
    ``window_length``. The average Jaccard dissimilarity to all other
    subsets in the pool is computed and the ``n_windows`` most diverse
    subsets are returned.  Windows are returned as lists of ``id`` values
    so the original row order can be restored later.
    """

    rng = np.random.default_rng(random_state)

    pools: Dict[str, Dict[int, List[List[str]]]] = {}
    for gid, grp in df.groupby(group_col):
        idxs = sorted(grp[id_col].astype(str).tolist())
        gid = str(gid)
        pools[gid] = {}
        for L in window_lengths:
            if len(idxs) < L:
                pools[gid][L] = []
                continue

            max_pool = math.comb(len(idxs), L)
            if max_pool <= N_pool:
                pool_windows = [list(c) for c in combinations(idxs, L)]
            else:
                seen = set()
                pool_windows = []
                while len(pool_windows) < N_pool:
                    cand = tuple(sorted(rng.choice(idxs, size=L, replace=False)))
                    if cand in seen:
                        continue
                    seen.add(cand)
                    pool_windows.append(list(cand))

            # precompute sets for Jaccard diversity
            sets = [set(w) for w in pool_windows]
            n = len(sets)
            if n == 1:
                diversity = np.array([0.0])
            else:
                diversity = np.zeros(n)
                for i in range(n):
                    acc = 0.0
                    for j in range(n):
                        if i == j:
                            continue
                        inter = len(sets[i] & sets[j])
                        union = len(sets[i] | sets[j])
                        sim = inter / union if union else 1.0
                        acc += 1.0 - sim
                    diversity[i] = acc / (n - 1)

            select_idx = np.argsort(-diversity)[:n_windows]
            pools[gid][L] = [pool_windows[i] for i in select_idx]

    return pools


def build_pairwise_comparisons(
    trails_per_animal: Dict[str, Dict[int, List[Tuple[int, List[str]]]]],
    df: pd.DataFrame,
    id_col: str,
    sex_map: Dict[str, str],
    fallback_map: Dict[str, bool],
    n_folds: int,
    random_state: int,
) -> Tuple[List[Dict], pd.DataFrame]:
    """Create cross- and within-individual comparisons and summary.

    ``samples_a`` and ``samples_b`` contain lists of ``id`` values
    corresponding to the original rows of ``df``.
    """

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
        # use string values to ensure consistent dtype across the column
        # (cross-individual comparisons use "True"/"False" strings)
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

    # --- assign folds by individual to avoid data leakage ---
    id_labels = [sex_map[i] for i in all_ids]
    skf_ind = StratifiedKFold(
        n_splits=n_folds, shuffle=True, random_state=random_state
    )
    id_to_fold: Dict[str, int] = {}
    for fold, (_, val_idx) in enumerate(skf_ind.split(all_ids, id_labels)):
        for vi in val_idx:
            id_to_fold[all_ids[vi]] = fold

    filtered: List[Dict] = []
    for comp in comparisons:
        fa = id_to_fold[comp["ind_a"]]
        fb = id_to_fold[comp["ind_b"]]
        if fa != fb:
            continue
        comp["fold"] = fa
        filtered.append(comp)

    comparisons = filtered

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
    # ``same_individual`` is stored as the strings "True", "False" or "unknown".
    # Count the occurrences accordingly for the summary statistics.
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


def generate_pairwise_comparisons_from_df(
    df: pd.DataFrame,
    *,
    id_col: str = "individual_id",
    sample_col: str = "id",
    group_sizes: List[int] | None = None,
    n_repeats: int = 5,
    mode: str = "both",
    selfmatch_factor: float = 2.0,
    n_folds: int = 5,
    random_state: int = 0,
    show_progress: bool = False,
) -> Tuple[List[Dict], pd.DataFrame]:
    """Wrapper for :func:`geometric_pairwise_projection.generate_pairwise_comparisons_from_df`."""

    comps = _generate_pw_from_df(
        df=df,
        id_col=id_col,
        sample_col=sample_col,
        group_sizes=group_sizes or [3, 5, 7, 10],
        n_repeats=n_repeats,
        mode=mode,
        selfmatch_factor=selfmatch_factor,
        n_folds=n_folds,
        random_state=random_state,
        show_progress=show_progress,
    )
    return comps, pd.DataFrame()


def generate_pairwise_comparisons_from_df(
    df: pd.DataFrame,
    *,
    id_col: str = "individual_id",
    sample_col: str = "id",
    group_sizes: List[int] = [3, 5, 7, 10],
    n_repeats: int = 5,
    mode: str = "both",
    selfmatch_factor: float = 2.0,
    n_folds: int = 5,
    random_state: int = 0,
    show_progress: bool = False,
) -> List[Dict]:
    """Create trail pair comparisons with metadata.

    Parameters
    ----------
    df:
        Input dataframe containing one row per footprint.
    id_col:
        Column that identifies the individual for grouping.
    sample_col:
        Column containing unique sample identifiers. ``samples_a`` and
        ``samples_b`` in the returned comparisons reference values from this
        column.

    Returns
    -------
    list of dict
        Each dictionary describes one comparison with trail IDs and fold
        assignment.
    """
    # 1) Alle Roh-Paare sammeln
    individuals = defaultdict(list)
    it = df.iterrows()
    if show_progress:
        it = tqdm(it, total=len(df), desc="index", leave=False)
    for _, row in it:
        individuals[row[id_col]].append(row[sample_col])

    raw = []
    for size_a in tqdm(
        group_sizes, desc="size_a", leave=False, disable=not show_progress
    ):
        for size_b in tqdm(
            group_sizes,
            desc=f"size_b({size_a})",
            leave=False,
            disable=not show_progress,
        ):
            if mode == "symmetric" and size_a != size_b:
                continue
            if mode == "asymmetric" and size_a == size_b:
                continue

            elig_a = [ind for ind, s in individuals.items() if len(s) >= size_a]
            elig_b = [ind for ind, s in individuals.items() if len(s) >= size_b]

            # cross-individual
            for ind_a in elig_a:
                for ind_b in elig_b:
                    if ind_a >= ind_b:
                        continue
                    for _ in range(n_repeats):
                        sa = random.sample(individuals[ind_a], size_a)
                        sb = random.sample(individuals[ind_b], size_b)
                        raw.append((ind_a, ind_b, sa, sb, False))

            # same-individual
            for ind in individuals:
                if len(individuals[ind]) < size_a + size_b:
                    continue
                for _ in range(int(n_repeats * selfmatch_factor)):
                    combo = random.sample(individuals[ind], size_a + size_b)
                    sa, sb = combo[:size_a], combo[size_a:]
                    raw.append((ind, ind, sa, sb, True))

    # 2) Stratified K-Fold auf Pair-Level (ind_a, ind_b)
    pair_keys, y = [], []
    seen = {}
    for ind_a, ind_b, sa, sb, same in raw:
        key = (ind_a, ind_b)
        if key not in seen:
            seen[key] = same
            pair_keys.append(key)
            y.append(1 if same else 0)

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    fold_map = {}
    for fold_idx, (_, val_idx) in enumerate(skf.split(pair_keys, y)):
        for pi in val_idx:
            fold_map[pair_keys[pi]] = fold_idx

    # 3) trail ID counter per individual and group size
    trail_counters = defaultdict(int)

    # 4) final list with trail IDs and fold assignment
    comparisons = []
    it_raw = raw
    if show_progress:
        it_raw = tqdm(raw, desc="pairs", leave=False)
    for ind_a, ind_b, sa, sb, same in it_raw:
        size_a = len(sa)
        trail_counters[(ind_a, size_a)] += 1
        letter_a = chr(ord("a") + (trail_counters[(ind_a, size_a)] - 1) % 26)
        trail_a_id = f"{ind_a}_{size_a}{letter_a}"

        size_b = len(sb)
        trail_counters[(ind_b, size_b)] += 1
        letter_b = chr(ord("a") + (trail_counters[(ind_b, size_b)] - 1) % 26)
        trail_b_id = f"{ind_b}_{size_b}{letter_b}"

        comparisons.append(
            {
                "ind_a": ind_a,
                "ind_b": ind_b,
                "samples_a": sa,
                "samples_b": sb,
                "trail_a_id": trail_a_id,
                "trail_b_id": trail_b_id,
                "same_individual": same,
                "fold": fold_map[(ind_a, ind_b)],
            }
        )

    return comparisons

#@ gtp this is the version I want to have implemented!"!! delete all other versions that do similar things in this py file 
def select_ or_generate_trails
    if trail colum is selected retuleren values from trail colum
    if genetaret trails is selected do this:
        
        Sample_size = 9 #can be changed
        for every unique individual df[individual_id] estimate in how many samples full samples can be created per individual.  (eg. individual with 30 observation gernerates 3 trails). Randomly sample without replacement for each individual;
        return trails  (update values in trail colum) naming: f "inidividual_id"_f(sample_size)_f(integer a, b, c....)  #Trails are either 

#@ gtp this is the version I want to have implemented!"!! delete all other versions that do similar things in this py file 
def generate subsamples of smaler trail Sample_size
    for every unique individual with rows Trail = NA distribute all rows equually between all trails of this individual to increase sample poolsize.
    for every unique value in trails generate subsamples of size [3,5,7]
    draw subsamples with replacement _n = 20 times per unique value in trail and sample Sample_size
    calculate jacard indey for all subsamles of one trail and one Sample_size
    selected number of subsamples = 3  by selecting 3 trails with lowest similarity per trail and subsample size; if JAquard index = 1 reduce number of subsamples for this trail. 
    map all subsamples to trail id and df["id"] so that features can be assigned later correctly
    return trails naming: f (name that was from select or generate trails )_ fsub(subsamplesize)_sample(subsamples)  

#@ gtp this is the version I want to have implemented!"!! delete all other versions that do similar things in this py file 
def generate pairs ()
    generate pairs of all trail combinations and all combinatiions of all subsample that do not originate from the same trail.



def run_all_pairwise_projections_parallel(
    comparisons: List[Dict],
    df: pd.DataFrame,
    feature_cols: List[str],
    k_features: Union[int, List[int]] = 15,
    reducers: List[str] = ["lda"],
    selection_method: str = "forward",
    n_components: Union[int, List[int]] = 2,
    use_sexmodel_prediction: bool = False,
    sexmodel_path: str = None,
    debug: bool = False,
    n_jobs: int = -1,
    batch_size: int | None = None,
) -> List[Dict]:
    """Run projections for all pairings.

    Steps
    -----
    0. If ``use_sexmodel_prediction`` is ``True``, load the sex model and precompute ``predict_proba`` on the entire base DataFrame.
    1. Clean the base DataFrame.
    2. Perform feature selection once with ``k_max``.
    3. Use an RCV set comprising the remaining indices.
    4. Extract ``predict_proba`` for A, B and R and compute their averages.
    5. For every combination of ``reducer``, ``n_components`` and ``k``:
       - apply the dimensionality reducer
       - compute distances
       - record results including averaged probabilities

    Parameters
    ----------
    batch_size : int or None, optional
        If set, comparisons are processed in batches of this size.
    """
    # 0) Sex-Modell
    if use_sexmodel_prediction:
        if not sexmodel_path:
            raise ValueError(
                "sexmodel_path must be provided when use_sexmodel_prediction=True"
            )
        sex_clf = load(sexmodel_path)

    # 1) Basis-DF
    df2 = df.copy()
    df2["id"] = df2["id"].astype(str)
    df2[feature_cols] = df2[feature_cols].apply(pd.to_numeric, errors="coerce")
    df_base = df2.set_index("id")

    # 2) predict_proba komplett vorberechnen
    if use_sexmodel_prediction:
        proba_all = pd.DataFrame(
            sex_clf.predict_proba(df_base[feature_cols]),
            index=df_base.index,
        )
    else:
        proba_all = None

    # 3) Parameter
    ks = k_features if isinstance(k_features, (list, tuple)) else [k_features]
    k_max = max(ks)
    ncs = n_components if isinstance(n_components, (list, tuple)) else [n_components]

    def process_pair(i: int, comp: Dict) -> List[Dict]:
        out = []
        ind_a, ind_b = comp["ind_a"], comp["ind_b"]
        ids_a = [str(i) for i in comp["samples_a"]]
        ids_b = [str(i) for i in comp["samples_b"]]
        size_a, size_b = len(ids_a), len(ids_b)

        trail_a_id = comp["trail_a_id"]
        trail_b_id = comp["trail_b_id"]

        # 1) feature matrices for A & B
        df_a = df_base.loc[ids_a, feature_cols]
        df_b = df_base.loc[ids_b, feature_cols]

        # 2) feature selection using ``k_max``
        X_ab = pd.concat([df_a, df_b], ignore_index=True)
        y_ab = np.concatenate([np.zeros(size_a, int), np.ones(size_b, int)])
        selector = FeatureSelectionTransformer(method=selection_method, k=k_max)
        selector.fit(X_ab, y_ab)
        full_ranking = selector.feature_ranking_

        # 3) RCV set as the complement
        rcv_ids = df_base.index.difference(ids_a + ids_b)
        df_r = df_base.loc[rcv_ids, feature_cols]

        # 4) extract and average sex probabilities per group
        if use_sexmodel_prediction:
            pa = proba_all.loc[ids_a].to_numpy()
            pb = proba_all.loc[ids_b].to_numpy()
            pr = proba_all.loc[rcv_ids].to_numpy()
            avg_A_0, avg_A_1 = float(pa[:, 0].mean()), float(pa[:, 1].mean())
            avg_B_0, avg_B_1 = float(pb[:, 0].mean()), float(pb[:, 1].mean())
            avg_R_0, avg_R_1 = float(pr[:, 0].mean()), float(pr[:, 1].mean())

            # 5) projection and distance
            for reducer in tqdm(reducers, desc=f"[Pair {i}] Reducer", leave=False):
                supervised = reducer in ("lda", "umap")
                for nc in tqdm(ncs, desc=f"[Pair {i} / {reducer}] n_comp", leave=False):
                    for k in tqdm(
                        ks, desc=f"[Pair {i} / {reducer} / nc={nc}] k", leave=False
                    ):
                        sel_feats = [feat for feat, _ in full_ranking[:k]]
                        da = df_a[sel_feats].copy()
                        db = df_b[sel_feats].copy()
                        dr = df_r[sel_feats].copy()

                        # 6) append sex probabilities as features
                        if use_sexmodel_prediction:
                            da["proba_0"], da["proba_1"] = pa[:, 0], pa[:, 1]
                            db["proba_0"], db["proba_1"] = pb[:, 0], pb[:, 1]
                            dr["proba_0"], dr["proba_1"] = pr[:, 0], pr[:, 1]

                        # 7) combine and build labeled array
                        arr = pd.concat([da, db, dr], ignore_index=True)
                        y_all = np.concatenate(
                            [
                                np.zeros(len(da), int),
                                np.ones(len(db), int),
                                np.full(len(dr), 2, int),
                            ]
                        )

                        # 8) clamp dimensions for LDA
                        nc_eff = nc
                        if reducer == "lda":
                            n_classes = len(np.unique(y_all))
                            nc_eff = min(nc, arr.shape[1], n_classes - 1)
                            if nc_eff < 1:
                                if debug:
                                    print(f"[DEBUG] skip LDA pair {i}, k={k}, nc={nc}")
                                continue

                        # 9) Fit & transform
                        dr_model = DimensionalityReducerTransformer(
                            method=reducer, n_components=nc_eff, supervised=supervised
                        )
                        dr_model.fit(arr, y_all if supervised else None)
                        coords = dr_model.transform(arr)

                        ca = coords[: len(da)]
                        cb = coords[len(da) : len(da) + len(db)]
                        cr = coords[len(da) + len(db) :]

                        cA, cB, cR = ca.mean(axis=0), cb.mean(axis=0), cr.mean(axis=0)
                        dists = compute_distances(cA, cB)

                        # 10) Result-Dict
                        res = {
                            "trail_a_id": trail_a_id,
                            "trail_b_id": trail_b_id,
                            "samples_a": ids_a,
                            "samples_b": ids_b,
                            "ind_a": ind_a,
                            "ind_b": ind_b,
                            "same_individual": comp["same_individual"],
                            "fold": comp["fold"],
                            "pipeline": (
                                f"{selection_method}_k{k}_{reducer}_nc{nc_eff}_"
                                f"{'sex_on' if use_sexmodel_prediction else 'sex_off'}"
                            ),
                            "selection_method": selection_method,
                            "reducer": reducer,
                            "k_features": k,
                            "n_components": nc_eff,
                            "comparison_id": i,
                            "min_observations": min(size_a, size_b),
                            "max_observations": max(size_a, size_b),
                            "use_sexmodel_prediction": use_sexmodel_prediction,
                            **(
                                {
                                    "avg_proba_A_0": avg_A_0,
                                    "avg_proba_A_1": avg_A_1,
                                    "avg_proba_B_0": avg_B_0,
                                    "avg_proba_B_1": avg_B_1,
                                    "avg_proba_R_0": avg_R_0,
                                    "avg_proba_R_1": avg_R_1,
                                }
                                if use_sexmodel_prediction
                                else {}
                            ),
                            "center_a_x": float(cA[0]) if cA.size > 0 else None,
                            "center_a_y": float(cA[1]) if cA.size > 1 else None,
                            "center_b_x": float(cB[0]) if cB.size > 0 else None,
                            "center_b_y": float(cB[1]) if cB.size > 1 else None,
                            "center_r_x": float(cR[0]) if cR.size > 0 else None,
                            "center_r_y": float(cR[1]) if cR.size > 1 else None,
                            "coords_a_x": ca[:, 0].tolist(),
                            "coords_a_y": ca[:, 1].tolist() if ca.shape[1] > 1 else [],
                            "coords_b_x": cb[:, 0].tolist(),
                            "coords_b_y": cb[:, 1].tolist() if cb.shape[1] > 1 else [],
                            "coords_r_x": cr[:, 0].tolist(),
                            "coords_r_y": cr[:, 1].tolist() if cr.shape[1] > 1 else [],
                            "selected_features": sel_feats,
                            "selected_scores": [score for _, score in full_ranking[:k]],
                        }
                        for m, v in dists.items():
                            res[f"dist_{m}"] = float(v)

                        out.append(res)

            return out

    cached_pair = memory.cache(process_pair)

    results: List[Dict] = []
    batches = (
        [
            comparisons[i : i + batch_size]
            for i in range(0, len(comparisons), batch_size)
        ]
        if batch_size
        else [comparisons]
    )
    offset = 0
    for batch in tqdm(batches, desc="batches", leave=False):
        with tqdm_joblib(tqdm(desc="Processing Pairs", total=len(batch), leave=False)):
            nested = Parallel(n_jobs=n_jobs)(
                delayed(cached_pair)(offset + i, comp) for i, comp in enumerate(batch)
            )
        results.extend(row for group in nested for row in group)
        offset += len(batch)

    return results



# --- Legacy implementation -------------------------------------------------
# The block below used to contain the original implementation of
# ``generate_pairwise_comparisons_from_df``. It has been kept for reference but
# is no longer executed. Commenting out the old code avoids syntax errors while
# preserving the historical context.
#
## def generate_pairwise_comparisons_from_df(
##     df: pd.DataFrame,
##     id_col: str = "individual_id",
##     trail_size_list: Optional[List[int]] = None,
##     num_individuals: Optional[int] = None,
##     strict_individuals: bool = False,
##     n_folds: int = 3,
##     random_state: int = 0,
##     fallback_col: str = "trail",
##     *,
##     sampling_mode: Literal["predefined", "window"] = "window",
##     trails_per_animal: Optional[Dict] = None,
##     window_lengths: Optional[List[int]] = None,
##     N_pool: int = 500,
##     n_windows: int = 5,
## ) -> Tuple[List[Dict], pd.DataFrame]:
##     """Generate trail pairs with metadata.
##
##     Steps
##     -----
##     1. Create ``group_id`` from ``id_col`` or ``fallback_col``.
##     2. Obtain ``trails_per_animal`` either from predefined pools or via
##        diverse subset sampling based on Jaccard dissimilarity (default).
##     3. Build cross- and within-individual pairings.
##     4. Mark ``same_individual`` and ``same_sex`` as boolean or ``"unknown"``.
##     5. Apply ``StratifiedKFold`` on ``trail_size_a``.
##     6. Return a summary table with per-length and total statistics.
##     """
##     ...  # Implementation removed for brevity
