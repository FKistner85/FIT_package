from itertools import combinations
from typing import Dict, List, Tuple, Union

import numpy as np
import pandas as pd

__all__ = [
    "select_or_generate_trails",
    "generate_subsamples",
    "generate_pairs",
    "run_all_pairwise_projections_parallel",
]



def select_or_generate_trails(
    df: pd.DataFrame,
    *,
    strategy: str = "generate",
    individual_col: str = "individual_id",
    trail_col: str = "trail",
    sample_size: int = 9,
    random_state: int = 0,
) -> pd.DataFrame:
    """Return existing trails or generate new ones.

    Parameters
    ----------
    df : pd.DataFrame
        Input data with one row per footprint.
    strategy : str, optional
        ``"select"`` to keep existing ``trail`` values or ``"generate``" to
        create new trails. Defaults to ``"generate"``.
    individual_col : str, optional
        Column identifying individuals. Defaults to ``"individual_id"``.
    trail_col : str, optional
        Column containing trail identifiers. Defaults to ``"trail"``.
    sample_size : int, optional
        Number of observations per generated trail. Defaults to ``9``.
    random_state : int, optional
        Seed for random sampling. Defaults to ``0``.

    Returns
    -------
    pd.DataFrame
        ``df`` with updated ``trail_col`` values.
    """

    df = df.copy()
    rng = np.random.default_rng(random_state)

    if strategy == "select" and trail_col in df.columns:
        return df

    df[trail_col] = pd.NA
    for ind, grp in df.groupby(individual_col):
        idxs = grp.index.tolist()
        rng.shuffle(idxs)
        n_trails = len(idxs) // sample_size
        for i in range(n_trails):
            sel = idxs[i * sample_size : (i + 1) * sample_size]
            trail_id = f"{ind}_{sample_size}_{i + 1}"
            df.loc[sel, trail_col] = trail_id

    return df


def generate_subsamples(
    df: pd.DataFrame,
    *,
    trail_col: str = "trail",
    id_col: str = "id",
    individual_col: str = "individual_id",
    sample_size: int = 9,
    subsample_sizes: Tuple[int, ...] = (3, 5, 7),
    n_candidates: int = 20,
    random_state: int = 0,
) -> Dict[str, List[str]]:
    """Create diverse subsamples for each trail."""

    df = df.copy()
    rng = np.random.default_rng(random_state)

    for ind, grp in df.groupby(individual_col):
        trails = grp[trail_col].dropna().unique().tolist()
        if not trails:
            continue
        remaining = grp[grp[trail_col].isna()].index.tolist()
        if remaining:
            assign = rng.choice(trails, size=len(remaining))
            for idx, tr in zip(remaining, assign):
                df.at[idx, trail_col] = tr

    trail_to_ids = {
        tr: df.loc[df[trail_col] == tr, id_col].tolist()
        for tr in df[trail_col].dropna().unique()
    }

    subsamples: Dict[str, List[str]] = {}
    for tr, ids in trail_to_ids.items():
        for ss in subsample_sizes:
            if len(ids) < ss:
                continue
            cand = [list(rng.choice(ids, size=ss, replace=False)) for _ in range(n_candidates)]
            sets = [set(c) for c in cand]
            sims = np.zeros(len(cand))
            for i, a in enumerate(sets):
                if len(cand) == 1:
                    sims[i] = 0.0
                else:
                    acc = 0.0
                    for j, b in enumerate(sets):
                        if i == j:
                            continue
                        inter = len(a & b)
                        union = len(a | b)
                        acc += inter / union if union else 1.0
                    sims[i] = acc / (len(cand) - 1)
            order = np.argsort(sims)
            chosen = []
            for idx in order:
                c = cand[idx]
                if any(set(c) == set(o) for o in chosen):
                    continue
                chosen.append(c)
                if len(chosen) == 3:
                    break
            for i, sub in enumerate(chosen, 1):
                name = f"{tr}_sub{ss}_sample{i}"
                subsamples[name] = sub

    return subsamples


def generate_pairs(trails: Dict[str, List[str]]) -> List[Tuple[str, str]]:
    """Create all trail pair combinations excluding same base trail."""

    names = list(trails.keys())
    pairs: List[Tuple[str, str]] = []
    for i in range(len(names)):
        base_i = names[i].split("_sub")[0]
        for j in range(i + 1, len(names)):
            base_j = names[j].split("_sub")[0]
            if base_i == base_j:
                continue
            pairs.append((names[i], names[j]))
    return pairs



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



