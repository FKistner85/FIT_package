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


import itertools
from collections import defaultdict
from typing import List, Dict, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

def generate_pairwise_comparisons_from_df(
    df: pd.DataFrame,
    id_col: str = "individual_id",
    chunk_size: int = 10,
    n_repeats: int = 1,
    n_folds: int = 5,
    random_state: int = 0
) -> Tuple[List[Dict], pd.DataFrame]:
    """
    1) Für jedes Individuum dessen indices in non-overlapping Chunks der Länge chunk_size
       (letzte Gruppe ggf. kleiner) aufteilen.
    2) Aus jedem Chunk ein Trail-Objekt erzeugen mit:
         { "ind": individual_id,
           "trail_id": f"{individual_id}_{len(chunk)}_{letter}",
           "samples": chunk_indices }
    3) StratifiedKFold auf TRAIL-Level nach individual_id
    4) Alle paarweisen Kombinationen aller Trails bilden, jedem Paar den Fold = max(fold_a, fold_b) zuweisen
    5) Summary-Tabelle pro chunk_size und gesamt zurückgeben
    """
    rng = np.random.default_rng(random_state)

    # 1) Gruppierung in Chunks
    individuals = defaultdict(list)
    for idx, row in df.iterrows():
        individuals[row[id_col]].append(idx)

    trails = []
    trail_counters = defaultdict(int)
    for ind, idx_list in individuals.items():
        # shuffle to randomize chunk assignment
        idx_list = idx_list.copy()
        rng.shuffle(idx_list)
        # split into chunks of size chunk_size
        for i in range(0, len(idx_list), chunk_size):
            chunk = idx_list[i:i+chunk_size]
            trail_counters[ind] += 1
            suffix = chr(ord('a') + (trail_counters[ind]-1)%26)
            trail_id = f"{ind}_{len(chunk)}{suffix}"
            trails.append({
                "ind": ind,
                "trail_id": trail_id,
                "samples": chunk
            })

    # 2) StratifiedKFold auf trail_id nach ind
    keys = [t["trail_id"] for t in trails]
    y = [t["ind"] for t in trails]
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    fold_map = {}
    for fold_idx, (_, val_idx) in enumerate(skf.split(keys, y)):
        for i in val_idx:
            fold_map[keys[i]] = fold_idx

    # 3) Paarweises Kombinieren aller Trails
    comparisons = []
    for a, b in itertools.combinations(trails, 2):
        fold = max(fold_map[a["trail_id"]], fold_map[b["trail_id"]])
        comparisons.append({
            "ind_a":           a["ind"],
            "ind_b":           b["ind"],
            "trail_a_id":      a["trail_id"],
            "trail_b_id":      b["trail_id"],
            "samples_a":       a["samples"],
            "samples_b":       b["samples"],
            "same_individual": (a["ind"] == b["ind"]),
            "fold":            fold
        })

    # 4) Summary
    df_trails = pd.DataFrame(trails)
    # per animal
    per_animal = df_trails.groupby("ind").agg(
        n_trails = ("trail_id","count"),
        avg_prints = ("samples", lambda x: np.mean([len(s) for s in x])),
        sd_prints = ("samples", lambda x: np.std([len(s) for s in x], ddof=1))
    ).reset_index()
    summary = per_animal.agg(
        number_of_animals = ("ind", "nunique"),
        number_of_trails = ("n_trails", "sum"),
        avg_trails_per_animal = ("n_trails", "mean"),
        sd_trails_per_animal = ("n_trails", lambda x: np.std(x, ddof=1)),
        avg_prints_per_animal = ("avg_prints", "mean"),
        sd_prints_per_animal = ("avg_prints", lambda x: np.std(x, ddof=1))
    ).to_frame().T
    # attach chunk_size
    summary.insert(0, "chunk_size", chunk_size)

    return comparisons, summary



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
    Für jede Paarung:
      0) Falls use_sexmodel_prediction, Sex-Modell laden & predict_proba auf ganzem df_base vorberechnen
      1) Basis-DF bereinigen
      2) Feature-Selection (einmal mit k_max)
      3) RCV-Set als Komplement der Indizes
      4) predict_proba für A, B, R extrahieren und mitteln (jeweils avg für 0/1)
      5) Für jede (reducer, n_components, k):
         - DimRed erzeugen
         - Abstände berechnen
         - Result-Dict inkl. avg_proba_A_0/1, avg_proba_B_0/1, avg_proba_R_0/1
    """
    # 0) Sex-Modell
    if use_sexmodel_prediction:
        if not sexmodel_path:
            raise ValueError("sexmodel_path must be provided when use_sexmodel_prediction=True")
        sex_clf = load(sexmodel_path)

    # 1) Basis-DF
    df2 = df.copy()
    df2[feature_cols] = df2[feature_cols].apply(pd.to_numeric, errors="coerce")
    df_base = df2.reset_index(drop=True)

    # 2) predict_proba komplett vorberechnen
    if use_sexmodel_prediction:
        proba_all = sex_clf.predict_proba(df_base[feature_cols])
    else:
        proba_all = None

    # 3) Parameter
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

            # 1) Feature‐Matrices für A & B
            df_a = df_base.loc[idx_a, feature_cols]
            df_b = df_base.loc[idx_b, feature_cols]

            # 2) Feature‐Selection mit k_max
            X_ab = pd.concat([df_a, df_b], ignore_index=True)
            y_ab = np.concatenate([np.zeros(size_a, int), np.ones(size_b, int)])
            selector = FeatureSelectionTransformer(method=selection_method, k=k_max)
            selector.fit(X_ab, y_ab)
            full_ranking = selector.feature_ranking_

            # 3) RCV‐Set als Komplement
            all_idx = np.arange(len(df_base))
            rcv_idx = list(set(all_idx) - set(idx_a) - set(idx_b))
            df_r = df_base.loc[rcv_idx, feature_cols]

            # 4) Sex‐Probas pro Gruppe extrahieren & mitteln
            if use_sexmodel_prediction:
                pa = proba_all[idx_a]   # shape (size_a,2)
                pb = proba_all[idx_b]   # shape (size_b,2)
                pr = proba_all[rcv_idx] # shape (len(rcv_idx),2)
                avg_A_0, avg_A_1 = float(pa[:,0].mean()), float(pa[:,1].mean())
                avg_B_0, avg_B_1 = float(pb[:,0].mean()), float(pb[:,1].mean())
                avg_R_0, avg_R_1 = float(pr[:,0].mean()), float(pr[:,1].mean())

            # 5) Projection & Distance
            for reducer in tqdm(reducers, desc=f"[Pair {i}] Reducer", leave=False):
                supervised = reducer in ("lda", "umap")
                for nc in tqdm(ncs, desc=f"[Pair {i} / {reducer}] n_comp", leave=False):
                    for k in tqdm(ks, desc=f"[Pair {i} / {reducer} / nc={nc}] k", leave=False):
                        sel_feats = [feat for feat,_ in full_ranking[:k]]
                        da = df_a[sel_feats].copy()
                        db = df_b[sel_feats].copy()
                        dr = df_r[sel_feats].copy()

                        # 6) Sex‐Probas als Features anhängen
                        if use_sexmodel_prediction:
                            da["proba_0"], da["proba_1"] = pa[:,0], pa[:,1]
                            db["proba_0"], db["proba_1"] = pb[:,0], pb[:,1]
                            dr["proba_0"], dr["proba_1"] = pr[:,0], pr[:,1]

                        # 7) Combine + Labeled array
                        arr = pd.concat([da, db, dr], ignore_index=True)
                        y_all = np.concatenate([
                            np.zeros(len(da), int),
                            np.ones(len(db), int),
                            np.full(len(dr), 2, int)
                        ])

                        # 8) Clamp für LDA
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

                        # 10) Result-Dict
                        res = {
                            "trail_a_id":        trail_a_id,
                            "trail_b_id":        trail_b_id,
                            "samples_a":         idx_a,
                            "samples_b":         idx_b,
                            "ind_a":             ind_a,
                            "ind_b":             ind_b,
                            "same_individual":   comp["same_individual"],
                            "fold":              comp["fold"],
                            "pipeline": (    f"{selection_method}_k{k}_{reducer}_nc{nc_eff}_"f"{'sex_on' if use_sexmodel_prediction else 'sex_off'}"),
                            "selection_method":  selection_method,
                            "reducer":           reducer,
                            "k_features":        k,
                            "n_components":      nc_eff,
                            "comparison_id":     i,
                            "min_observations":  min(size_a, size_b),
                            "max_observations":  max(size_a, size_b),
                            "use_sexmodel_prediction": use_sexmodel_prediction,
                            **({"avg_proba_A_0": avg_A_0,
                                "avg_proba_A_1": avg_A_1,
                                "avg_proba_B_0": avg_B_0,
                                "avg_proba_B_1": avg_B_1,
                                "avg_proba_R_0": avg_R_0,
                                "avg_proba_R_1": avg_R_1}
                               if use_sexmodel_prediction else {}),
                            "center_a_x":  float(cA[0]) if cA.size>0 else None,
                            "center_a_y":  float(cA[1]) if cA.size>1 else None,
                            "center_b_x":  float(cB[0]) if cB.size>0 else None,
                            "center_b_y":  float(cB[1]) if cB.size>1 else None,
                            "center_r_x":  float(cR[0]) if cR.size>0 else None,
                            "center_r_y":  float(cR[1]) if cR.size>1 else None,
                            "coords_a_x": ca[:,0].tolist(),
                            "coords_a_y": ca[:,1].tolist() if ca.shape[1]>1 else [],
                            "coords_b_x": cb[:,0].tolist(),
                            "coords_b_y": cb[:,1].tolist() if cb.shape[1]>1 else [],
                            "coords_r_x": cr[:,0].tolist(),
                            "coords_r_y": cr[:,1].tolist() if cr.shape[1]>1 else [],
                        }
                        for m, v in dists.items():
                            res[f"dist_{m}"] = float(v)

                        out.append(res)

            return out

        except Exception as e:
            if debug:
                print(f"[ERROR] pair {i} failed: {e}")
            return []

    # 11) Parallel-Ausführung mit äußerem Fortschritt
    with tqdm_joblib(tqdm(desc="Processing Pairs", total=len(comparisons))):
        nested = Parallel(n_jobs=n_jobs)(
            delayed(process_pair)(i, comp)
            for i, comp in enumerate(comparisons)
        )

    # flatten und zurückgeben
    return [row for group in nested for row in group]
