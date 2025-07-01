from typing import List, Dict, Union
import random

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from joblib import Parallel, delayed
from tqdm import tqdm

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
    Erzeuge Paarvergleiche und teile sie stratifiziert in n_folds auf.
    Jedes Element enthält:
      - ind_a, ind_b, size_a, size_b, samples_a, samples_b, same_individual, fold
    """
    # 1) Sammle alle raw Paare
    individuals = {}
    for idx, row in df.iterrows():
        individuals.setdefault(row[id_col], []).append(idx)

    raw = []
    for size_a in group_sizes:
        for size_b in group_sizes:
            if mode == 'symmetric' and size_a != size_b:
                continue
            if mode == 'asymmetric' and size_a == size_b:
                continue

            elig_a = [ind for ind, s in individuals.items() if len(s) >= size_a]
            elig_b = [ind for ind, s in individuals.items() if len(s) >= size_b]

            # Cross-individual
            for ind_a in elig_a:
                for ind_b in elig_b:
                    if ind_a >= ind_b:
                        continue
                    for _ in range(n_repeats):
                        sa = random.sample(individuals[ind_a], size_a)
                        sb = random.sample(individuals[ind_b], size_b)
                        raw.append((ind_a, ind_b, sa, sb, False))

            # Same-individual
            for ind in individuals:
                if len(individuals[ind]) < size_a + size_b:
                    continue
                for _ in range(int(n_repeats * selfmatch_factor)):
                    combo = random.sample(individuals[ind], size_a + size_b)
                    raw.append((ind, ind, combo[:size_a], combo[size_a:], True))

    # 2) Erstelle Unique-Paar-Keys für Stratifikation
    unique = {}
    for ind_a, ind_b, _, _, same in raw:
        key = (ind_a, ind_b)
        unique[key] = same

    pair_keys = list(unique.keys())
    y = [1 if unique[k] else 0 for k in pair_keys]

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    fold_map = {}
    for fold_idx, (_, val_idx) in enumerate(skf.split(pair_keys, y)):
        for pi in val_idx:
            fold_map[pair_keys[pi]] = fold_idx

    # 3) Baue finale Vergleichsliste mit Fold-Zuweisung
    comparisons = []
    for ind_a, ind_b, sa, sb, same in raw:
        comparisons.append({
            "ind_a": ind_a,
            "ind_b": ind_b,
            "size_a": len(sa),
            "size_b": len(sb),
            "samples_a": sa,
            "samples_b": sb,
            "same_individual": same,
            "fold": fold_map[(ind_a, ind_b)]
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
    """
    Für jede Paarung:
      1) Feature-Selection einmal mit k_max
      2) Für jede Kombination von (reducer, n_components, k) Projektion & Distanz
      3) Ergebnisse als Dict-Liste zurückgeben
    """
    # 1) Parameter als Listen
    ks = k_features if isinstance(k_features, (list, tuple)) else [k_features]
    k_max = max(ks)
    ncs = n_components if isinstance(n_components, (list, tuple)) else [n_components]

    # 2) Bereite Basis-DataFrame vor
    df_work = df.copy()
    df_work[feature_cols] = df_work[feature_cols].apply(pd.to_numeric, errors="coerce")
    df_base = df_work.reset_index(drop=True)

    def process_pair(i: int, comp: Dict) -> List[Dict]:
        try:
            idx_a, idx_b = comp["samples_a"], comp["samples_b"]
            df_a = df_base.loc[idx_a, feature_cols]
            df_b = df_base.loc[idx_b, feature_cols]

            # STEP 1: Feature-Selection einmal mit k_max
            X_ab = pd.concat([df_a, df_b], ignore_index=True)
            y_ab = np.concatenate([np.zeros(len(df_a), int), np.ones(len(df_b), int)])
            selector = FeatureSelectionTransformer(method=selection_method, k=k_max)
            selector.fit(X_ab, y_ab)
            full_ranking = selector.feature_ranking_

            # RCV-Daten
            df_r_full = generate_rcv(df_base, idx_a + idx_b)

            out = []
            # STEP 2: Loop über alle Reducer, n_components und k
            for reducer in reducers:
                supervised = reducer in ("lda", "umap")
                for nc in ncs:
                    for k in ks:
                        sel_feats = [feat for feat, _ in full_ranking[:k]]
                        da, db, dr = df_a[sel_feats], df_b[sel_feats], df_r_full[sel_feats]

                        arr = pd.concat([da, db, dr], ignore_index=True)
                        y_all = np.concatenate([
                            np.zeros(len(da), int),
                            np.ones(len(db), int),
                            np.full(len(dr), 2, int)
                        ])

                        # fit & transform
                        dr_model = DimensionalityReducerTransformer(
                            method=reducer,
                            n_components=nc,
                            supervised=supervised
                        )
                        dr_model.fit(arr, y_all if supervised else None)
                        coords = dr_model.transform(arr)

                        ca = coords[:len(da)]
                        cb = coords[len(da):len(da)+len(db)]
                        cr = coords[len(da)+len(db):]

                        cA, cB, cR = ca.mean(axis=0), cb.mean(axis=0), cr.mean(axis=0)
                        dists = compute_distances(cA, cB)

                        # Roh-Dict zusammenstellen
                        res = {
                            "ind_a": comp["ind_a"],
                            "ind_b": comp["ind_b"],
                            "same_individual": comp["same_individual"],
                            "fold": comp.get("fold"),
                            "pipeline": f"{selection_method}_k{k}_{reducer}_nc{nc}",
                            "selection_method": selection_method,
                            "reducer": reducer,
                            "k_features": k,
                            "n_components": nc,
                            "comparison_id": i,
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
                            **{f"dist_{m}": dists[m] for m in dists},
                            "min_observations": min(len(idx_a), len(idx_b)),
                            "max_observations": max(len(idx_a), len(idx_b)),
                        }

                        # STEP 3: np.* → native Python-Typen
                        clean = {}
                        for key, val in res.items():
                            if isinstance(val, np.generic):
                                clean[key] = val.item()
                            else:
                                clean[key] = val
                        out.append(clean)

            return out

        except Exception as e:
            if debug:
                print(f"[ERROR] pair {i} failed: {e}")
            return []

    # Parallel über alle Paare
    nested = Parallel(n_jobs=n_jobs)(
        delayed(process_pair)(i, comp)
        for i, comp in tqdm(enumerate(comparisons),
                            total=len(comparisons),
                            desc="Processing Pairs")
    )
    # flatten
    return [r for sub in nested for r in sub]
