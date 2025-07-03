from typing import List, Dict, Union
import numpy as np
import pandas as pd
from joblib import Parallel, delayed, load
from tqdm import tqdm
from tqdm_joblib import tqdm_joblib
from collections import defaultdict
import random
from sklearn.model_selection import StratifiedKFold
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
        size_a = len(sa)
        trail_counters[(ind_a, size_a)] += 1
        letter_a = chr(ord('a') + (trail_counters[(ind_a, size_a)] - 1) % 26)
        trail_a_id = f"{ind_a}_{size_a}{letter_a}"

        size_b = len(sb)
        trail_counters[(ind_b, size_b)] += 1
        letter_b = chr(ord('a') + (trail_counters[(ind_b, size_b)] - 1) % 26)
        trail_b_id = f"{ind_b}_{size_b}{letter_b}"

        comparisons.append({
            "ind_a":           ind_a,
            "ind_b":           ind_b,
            "samples_a":       sa,
            "samples_b":       sb,
            "trail_a_id":      trail_a_id,
            "trail_b_id":      trail_b_id,
            "same_individual": same,
            "fold":            fold_map[(ind_a, ind_b)]
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
    use_sexmodel_prediction: bool = False,
    sexmodel_path: str = None,
    debug: bool = False,
    n_jobs: int = -1
) -> List[Dict]:
    """
    Wie gehabt, plus:
    - use_sexmodel_prediction: wenn True, hängt nach FS noch die beiden
      predict_proba-Spalten (Klasse 0 & 1) an und gibt avg_proba_<0|1> sowie
      Listen probas_a/b/r mit aus.
    - sexmodel_path: Pfad zur .joblib-Datei des Sex-Modells (muss gesetzt sein,
      wenn use_sexmodel_prediction=True).
    """

    # 0) Sexmodell laden
    if use_sexmodel_prediction:
        if not sexmodel_path:
            raise ValueError("sexmodel_path must be provided when use_sexmodel_prediction=True")
        sex_clf = load(sexmodel_path)

    # 1) Basis-DF vorbereiten
    df2 = df.copy()
    df2[feature_cols] = df2[feature_cols].apply(pd.to_numeric, errors="coerce")
    df_base = df2.reset_index(drop=True)

    # 2) Predict_proba einmal auf gesamten Datensatz
    if use_sexmodel_prediction:
        proba_all = sex_clf.predict_proba(df_base[feature_cols])
    else:
        proba_all = None

    # 3) Parameter-Listen
    ks    = k_features if isinstance(k_features, (list, tuple)) else [k_features]
    k_max = max(ks)
    ncs   = n_components if isinstance(n_components, (list, tuple)) else [n_components]

    def process_pair(i: int, comp: Dict) -> List[Dict]:
        out = []
        try:
            ind_a, ind_b   = comp["ind_a"], comp["ind_b"]
            idx_a, idx_b   = comp["samples_a"], comp["samples_b"]
            size_a, size_b = len(idx_a), len(idx_b)

            trail_a_id = comp["trail_a_id"]
            trail_b_id = comp["trail_b_id"]

            # Feature‐Matrices
            df_a = df_base.loc[idx_a, feature_cols]
            df_b = df_base.loc[idx_b, feature_cols]

            # 1) Feature‐Selection mit k_max
            X_ab = pd.concat([df_a, df_b], ignore_index=True)
            y_ab = np.concatenate([np.zeros(size_a, int), np.ones(size_b, int)])
            selector = FeatureSelectionTransformer(method=selection_method, k=k_max)
            selector.fit(X_ab, y_ab)
            full_ranking = selector.feature_ranking_

            # 2) RCV‐Set
            df_r_full = generate_rcv(df_base, idx_a + idx_b)

            # 3) Sex‐Probas pro Paar extrahieren und mitteln
            if use_sexmodel_prediction:
                pa = proba_all[idx_a]            # (size_a, 2)
                pb = proba_all[idx_b]            # (size_b, 2)
                pr = proba_all[idx_a + idx_b]    # concatenated indices
                avg0 = float(np.concatenate([pa[:,0], pb[:,0], pr[:,0]]).mean())
                avg1 = float(np.concatenate([pa[:,1], pb[:,1], pr[:,1]]).mean())

            # 4) Projection & Distance für jede Kombination
            for reducer in tqdm(reducers, desc=f"[Pair {i}] Reducer", leave=False):
                supervised = reducer in ("lda", "umap")
                for nc in tqdm(ncs, desc=f"[Pair {i} / {reducer}] n_comp", leave=False):
                    for k in tqdm(ks, desc=f"[Pair {i} / {reducer} / nc={nc}] k", leave=False):
                        sel_feats = [feat for feat,_ in full_ranking[:k]]
                        da = df_a[sel_feats].copy()
                        db = df_b[sel_feats].copy()
                        dr = df_r_full[sel_feats].copy()

                        # Sex‐Probas als Features anhängen
                        if use_sexmodel_prediction:
                            da["proba_0"], da["proba_1"] = pa[:,0], pa[:,1]
                            db["proba_0"], db["proba_1"] = pb[:,0], pb[:,1]
                            dr["proba_0"], dr["proba_1"] = pr[:,0], pr[:,1]

                        # Combine and labels
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
                            nc_eff = min(nc, arr.shape[1], n_classes - 1)
                            if nc_eff < 1:
                                if debug:
                                    print(f"[DEBUG] skip LDA pair {i} k={k} nc={nc}")
                                continue

                        # Fit & transform
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
                            "trail_a_id":              trail_a_id,
                            "trail_b_id":              trail_b_id,
                            "samples_a":               idx_a,
                            "samples_b":               idx_b,
                            "ind_a":                   ind_a,
                            "ind_b":                   ind_b,
                            "same_individual":         comp["same_individual"],
                            "fold":                    comp.get("fold"),
                            "pipeline":                f"{selection_method}_k{k}_{reducer}_nc{nc_eff}",
                            "selection_method":        selection_method,
                            "reducer":                 reducer,
                            "k_features":              k,
                            "n_components":            nc_eff,
                            "comparison_id":           i,
                            "min_observations":        min(size_a, size_b),
                            "max_observations":        max(size_a, size_b),
                            "use_sexmodel_prediction": use_sexmodel_prediction,
                            **({"avg_proba_0": avg0, "avg_proba_1": avg1} if use_sexmodel_prediction else {}),
                            "center_a_x":              float(cA[0]) if cA.size>0 else None,
                            "center_a_y":              float(cA[1]) if cA.size>1 else None,
                            "center_b_x":              float(cB[0]) if cB.size>0 else None,
                            "center_b_y":              float(cB[1]) if cB.size>1 else None,
                            "center_r_x":              float(cR[0]) if cR.size>0 else None,
                            "center_r_y":              float(cR[1]) if cR.size>1 else None,
                            "coords_a_x":              ca[:,0].tolist(),
                            "coords_a_y":              ca[:,1].tolist() if ca.shape[1]>1 else [],
                            "coords_b_x":              cb[:,0].tolist(),
                            "coords_b_y":              cb[:,1].tolist() if cb.shape[1]>1 else [],
                            "coords_r_x":              cr[:,0].tolist(),
                            "coords_r_y":              cr[:,1].tolist() if cr.shape[1]>1 else [],
                        }

                        for m, v in dists.items():
                            res[f"dist_{m}"] = float(v) if not np.isnan(v) else None

                        out.append(res)

            return out

        except Exception as e:
            if debug:
                print(f"[ERROR] pair {i} failed: {e}")
            return []

    # 5) Parallel-Ausführung mit äußerem Fortschritt
    with tqdm_joblib(tqdm(desc="Processing Pairs", total=len(comparisons))):
        nested = Parallel(n_jobs=n_jobs)(
            delayed(process_pair)(i, comp)
            for i, comp in enumerate(comparisons)
        )

    # flatten
    return [row for group in nested for row in group]