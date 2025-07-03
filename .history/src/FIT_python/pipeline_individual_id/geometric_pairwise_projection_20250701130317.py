from typing import List, Dict, Union
import pandas as pd
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
from typing import List, Dict
from collections import defaultdict
import random
from FIT_python.pipeline_individual_id.rcv_sampling import generate_rcv
from FIT_python.pipeline_individual_id.feature_selection_wrapper import FeatureSelectionTransformer
from FIT_python.pipeline_individual_id.dimensionality_reduction_wrapper import DimensionalityReducerTransformer
from FIT_python.pipeline_individual_id.distance_metrics import compute_distances


def generate_pairwise_comparisons_from_df(
    df: pd.DataFrame,
    id_col: str = "individual_id",
    group_sizes: List[int] = [3, 5, 7, 10],
    n_repeats: int = 5,
    mode: str = 'both',
    selfmatch_factor: float = 2.0,
    n_folds: int = 5,
    random_state: int = 0
) -> List[Dict]:
    """
    Erzeuge Paarvergleiche und teile sie in n_folds ein (stratifiziert nach same_individual).

    Gibt List[Dict] zurück, jedes Dict enthält zusätzlich:
      - 'fold': int von 0..n_folds-1
    """
    # 1) Alle Paar-Instanzen sammeln
    individuals = defaultdict(list)
    for idx, row in df.iterrows():
        individuals[row[id_col]].append(idx)

    raw = []
    for size_a in group_sizes:
        for size_b in group_sizes:
            if mode == 'symmetric' and size_a != size_b:
                continue
            if mode == 'asymmetric' and size_a == size_b:
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

    # 2) Unique-Pair-Level für Stratification
    #    Wir möchten sicherstellen, dass jede (ind_a, ind_b) mit seinem same-Flag
    #    gleichmäßig über die Folds verteilt wird.
    unique = {}
    for ind_a, ind_b, sa, sb, same in raw:
        # key (a,b,same) – hier reicht (a,b) da same=True nur, wenn a==b
        key = (ind_a, ind_b)
        unique[key] = same

    pair_keys = list(unique.keys())
    y = [1 if unique[k] else 0 for k in pair_keys]  # Label für Stratifikation

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    fold_map = {}
    for fold_idx, (_, val_idx) in enumerate(skf.split(pair_keys, y)):
        for pi in val_idx:
            fold_map[pair_keys[pi]] = fold_idx

    # 3) Endgültige Liste mit Fold-Zuweisung
    comparisons = []
    for ind_a, ind_b, sa, sb, same in raw:
        fold = fold_map[(ind_a, ind_b)]
        comparisons.append({
            "ind_a": ind_a,
            "ind_b": ind_b,
            "size_a": len(sa),
            "size_b": len(sb),
            "samples_a": sa,
            "samples_b": sb,
            "same_individual": same,
            "fold": fold
        })

    return comparisons


def run_all_pairwise_projections_parallel(
    comparisons: List[Dict],
    df: pd.DataFrame,
    feature_cols: List[str],
    k_features: Union[int, List[int]] = 15,
    reducers: List[str] = ["lda"],
    selection_method: str = "forward",
    n_components: Union[int, List[int]] = 2,
    debug: bool = False,
    n_jobs: int = -1
) -> List[Dict]:
    # normalize k_features and n_components to lists
    ks = sorted(k_features) if isinstance(k_features, (list, tuple)) else [k_features]
    k_max = max(ks)
    ncs = sorted(n_components) if isinstance(n_components, (list, tuple)) else [n_components]

    # prepare DataFrame
    df_base = df.copy().reset_index(drop=True)
    df_base[feature_cols] = df_base[feature_cols].apply(pd.to_numeric, errors="coerce")

    def process_pair(i, comp):
        try:
            idx_a, idx_b = comp["samples_a"], comp["samples_b"]
            df_a = df_base.loc[idx_a, feature_cols]
            df_b = df_base.loc[idx_b, feature_cols]
            X_ab = pd.concat([df_a, df_b], ignore_index=True)
            y_ab = np.concatenate([np.zeros(len(df_a), int), np.ones(len(df_b), int)])

            # STEP 1: feature selection once with k_max
            selector = FeatureSelectionTransformer(method=selection_method, k=k_max)
            selector.fit(X_ab, y_ab)
            full_ranking = selector.feature_ranking_

            # prepare RCV
            df_r_full = generate_rcv(df_base, idx_a + idx_b)

            results = []
            for reducer in reducers:
                supervised = reducer in ("lda", "umap")
                for nc in ncs:
                    dr = DimensionalityReducerTransformer(
                        method=reducer,
                        n_components=nc,
                        supervised=supervised
                    )
                    for k in ks:
                        sel = [f for f, _ in full_ranking[:k]]
                        da = df_a[sel]
                        db = df_b[sel]
                        dr_full = df_r_full[sel]

                        # STEP 2: fit & transform
                        arr = pd.concat([da, db, dr_full], ignore_index=True)
                        y_all = np.concatenate([
                            np.zeros(len(da), int),
                            np.ones(len(db), int),
                            np.full(len(dr_full), 2, int)
                        ])
                        dr.fit(arr, y_all if supervised else None)
                        coords = dr.transform(arr)

                        ca = coords[:len(da)]
                        cb = coords[len(da):len(da)+len(db)]
                        cr = coords[len(da)+len(db):]

                        cA, cB, cR = ca.mean(0), cb.mean(0), cr.mean(0)
                        d = compute_distances(cA, cB)

                        results.append({
                            "ind_a": comp["ind_a"],
                            "ind_b": comp["ind_b"],
                            "same_individual": comp["same_individual"],
                            "fold": comp["fold"],                    # <— hier hinzugefügt für spätere Modellierung
                            # Gruppierte Pipeline-Info
                            "pipeline": f"{selection_method}_k{k}_{reducer}_nc{nc}",

                            "selection_method": selection_method,
                            "reducer":        reducer,
                            "k_features":     k,
                            "n_components":   nc,
                            "comparison_id":  i,
                            "center_a_x": cA[0] if cA.size>0 else None,
                            "center_a_y": cA[1] if cA.size>1 else None,
                            "center_b_x": cB[0] if cB.size>0 else None,
                            "center_b_y": cB[1] if cB.size>1 else None,
                            "center_r_x": cR[0] if cR.size>0 else None,
                            "center_r_y": cR[1] if cR.size>1 else None,
                            "coords_a_x": ca[:,0].tolist() if ca.shape[1]>0 else [],
                            "coords_a_y": ca[:,1].tolist() if ca.shape[1]>1 else [],
                            "coords_b_x": cb[:,0].tolist() if cb.shape[1]>0 else [],
                            "coords_b_y": cb[:,1].tolist() if cb.shape[1]>1 else [],
                            "coords_r_x": cr[:,0].tolist() if cr.shape[1]>0 else [],
                            "coords_r_y": cr[:,1].tolist() if cr.shape[1]>1 else [],
                            **{f"dist_{m}": d[m] for m in d},
                            "min_observations": min(len(idx_a), len(idx_b)),
                            "max_observations": max(len(idx_a), len(idx_b)),
                        })
            return results

        except Exception as e:
            if debug:
                print(f"[ERROR] pair {i} failed: {e}")
            return []

    # parallel over pairs
    all_lists = Parallel(n_jobs=n_jobs)(
        delayed(process_pair)(i, comp)
        for i, comp in tqdm(
            enumerate(comparisons),
            total=len(comparisons),
            desc="Processing Pairs"
        )
    )
    # flatten
    return [r for sub in all_lists for r in sub]
