from typing import List, Dict, Union
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold

from FIT_python.pipeline_individual_id.rcv_sampling import generate_rcv
from FIT_python.pipeline_individual_id.feature_selection_wrapper import FeatureSelectionTransformer
from FIT_python.pipeline_individual_id.dimensionality_reduction_wrapper import DimensionalityReducerTransformer
from FIT_python.pipeline_individual_id.distance_metrics import compute_distances


from typing import List, Dict
import pandas as pd
from collections import defaultdict
import random
from sklearn.model_selection import StratifiedKFold

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
    Erzeuge Trail-Paarvergleiche mit:
      - 'samples_a'/'samples_b': List[int] der Original-Indizes
      - 'trail_a_id'/'trail_b_id': z.B. "rosie_7a", "otella_8b", …
      - 'same_individual': bool
      - 'fold': Stratified k-fold nach same_individual
    """
    # 1) Alle Roh-Paare sammeln
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

    # 3) Trail-ID-Zähler pro (Individuum, Gruppengröße)
    trail_counters = defaultdict(int)

    # 4) Finale Liste mit Trail-IDs und Fold
    comparisons = []
    for ind_a, ind_b, sa, sb, same in raw:
        # Trail A
        size_a = len(sa)
        trail_counters[(ind_a, size_a)] += 1
        letter_a = chr(ord('a') + (trail_counters[(ind_a, size_a)] - 1) % 26)
        trail_a_id = f"{ind_a}_{size_a}{letter_a}"

        # Trail B
        size_b = len(sb)
        trail_counters[(ind_b, size_b)] += 1
        letter_b = chr(ord('a') + (trail_counters[(ind_b, size_b)] - 1) % 26)
        trail_b_id = f"{ind_b}_{size_b}{letter_b}"

        comparisons.append({
            "ind_a": ind_a,
            "ind_b": ind_b,
            "samples_a": sa,
            "samples_b": sb,
            "trail_a_id": trail_a_id,
            "trail_b_id": trail_b_id,
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
      1) Einmal Feature-Selection mit k_max
      2) Für jede Kombination (reducer, n_components, k) → Projektion + Distanz
      3) LDA: n_components_eff = min(requested, n_features, n_classes-1)
      4) Fügt trail_a_id, trail_b_id, samples_a und samples_b ins Result-Dict ein.
    """
    # 1) Parameter in Listen
    ks    = k_features if isinstance(k_features, (list, tuple)) else [k_features]
    k_max = max(ks)
    ncs   = n_components if isinstance(n_components, (list, tuple)) else [n_components]

    # 2) Basis-DF vorbereiten
    df2 = df.copy()
    df2[feature_cols] = df2[feature_cols].apply(pd.to_numeric, errors="coerce")
    df_base = df2.reset_index(drop=True)

    def process_pair(i: int, comp: Dict) -> List[Dict]:
        out = []
        try:
            ind_a, ind_b       = comp["ind_a"], comp["ind_b"]
            idx_a, idx_b       = comp["samples_a"], comp["samples_b"]
            size_a, size_b     = len(idx_a), len(idx_b)

            # Trail-IDs erzeugen
            trail_a_id = f"{ind_a}_{size_a}_{i}_A"
            trail_b_id = f"{ind_b}_{size_b}_{i}_B"

            # Samples direkt mitgeben
            samples_a = idx_a.copy()
            samples_b = idx_b.copy()

            df_a = df_base.loc[idx_a, feature_cols]
            df_b = df_base.loc[idx_b, feature_cols]

            # 1) Feature-Selection mit k_max
            X_ab = pd.concat([df_a, df_b], ignore_index=True)
            y_ab = np.concatenate([np.zeros(size_a, int), np.ones(size_b, int)])
            selector = FeatureSelectionTransformer(method=selection_method, k=k_max)
            selector.fit(X_ab, y_ab)
            full_ranking = selector.feature_ranking_

            # RCV-Set
            df_r_full = generate_rcv(df_base, idx_a + idx_b)

            # 2) Für jede Kombination (reducer, nc, k)
            for reducer in reducers:
                supervised = reducer in ("lda", "umap")
                for nc in ncs:
                    for k in ks:
                        sel_feats = [f for f,_ in full_ranking[:k]]
                        da, db, dr = df_a[sel_feats], df_b[sel_feats], df_r_full[sel_feats]

                        arr = pd.concat([da, db, dr], ignore_index=True)
                        y_all = np.concatenate([
                            np.zeros(len(da), int),
                            np.ones(len(db), int),
                            np.full(len(dr), 2, int)
                        ])

                        # Clamp n_components für LDA
                        nc_eff = nc
                        if reducer == "lda":
                            n_classes = len(np.unique(y_all))
                            nc_eff = min(nc, len(sel_feats), n_classes - 1)
                            if nc_eff < 1:
                                if debug:
                                    print(f"[DEBUG] skip LDA pair {i} k={k} nc={nc}")
                                continue

                        dr_model = DimensionalityReducerTransformer(
                            method=reducer,
                            n_components=nc_eff,
                            supervised=supervised
                        )
                        dr_model.fit(arr, y_all if supervised else None)
                        coords = dr_model.transform(arr)

                        ca = coords[:len(da)]
                        cb = coords[len(da):len(da)+len(db)]
                        cr = coords[len(da)+len(db):]

                        cA, cB, cR = ca.mean(axis=0), cb.mean(axis=0), cr.mean(axis=0)
                        dists = compute_distances(cA, cB)

                        # Result-Dict zusammenbauen
                        res = {
                            "trail_a_id":      trail_a_id,
                            "trail_b_id":      trail_b_id,
                            "samples_a":       samples_a,
                            "samples_b":       samples_b,
                            "ind_a":           ind_a,
                            "ind_b":           ind_b,
                            "same_individual": comp["same_individual"],
                            "fold":            comp.get("fold"),
                            "pipeline":        f"{selection_method}_k{k}_{reducer}_nc{nc_eff}",
                            "selection_method":selection_method,
                            "reducer":         reducer,
                            "k_features":      k,
                            "n_components":    nc_eff,
                            "comparison_id":   i,
                            "min_observations":min(size_a, size_b),
                            "max_observations":max(size_a, size_b),
                            "center_a_x":      float(cA[0]) if cA.size>0 else None,
                            "center_a_y":      float(cA[1]) if cA.size>1 else None,
                            "center_b_x":      float(cB[0]) if cB.size>0 else None,
                            "center_b_y":      float(cB[1]) if cB.size>1 else None,
                            "center_r_x":      float(cR[0]) if cR.size>0 else None,
                            "center_r_y":      float(cR[1]) if cR.size>1 else None,
                            "coords_a_x":      ca[:,0].tolist(),
                            "coords_a_y":      ca[:,1].tolist() if ca.shape[1]>1 else [],
                            "coords_b_x":      cb[:,0].tolist(),
                            "coords_b_y":      cb[:,1].tolist() if cb.shape[1]>1 else [],
                            "coords_r_x":      cr[:,0].tolist(),
                            "coords_r_y":      cr[:,1].tolist() if cr.shape[1]>1 else [],
                        }

                        # Distanzen hinzufügen
                        for m, v in dists.items():
                            res[f"dist_{m}"] = float(v) if not np.isnan(v) else None

                        out.append(res)

            return out

        except Exception as e:
            if debug:
                print(f"[ERROR] pair {i} failed: {e}")
            return []

    # 4) Parallel über alle Paare
    nested = Parallel(n_jobs=n_jobs)(
        delayed(process_pair)(i, comp)
        for i, comp in tqdm(enumerate(comparisons),
                            total=len(comparisons),
                            desc="Processing Pairs")
    )
    # flatten
    return [row for group in nested for row in group]