# src/FIT_python/pipeline_individual_id/geometric_pairwise_projection.py

# Standard library
import random
from collections import defaultdict
from itertools import combinations
from pathlib import Path


# Third-party
import numpy as np
import pandas as pd
from joblib import Parallel, delayed, load
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
from tqdm_joblib import tqdm_joblib

from FIT_python.config import RESULTS_DATA_DIR

# Lokale Module
from FIT_python.pipeline_individual_id.rcv_sampling import generate_rcv
from FIT_python.pipeline_individual_id.feature_selection_wrapper import FeatureSelectionTransformer
from FIT_python.pipeline_individual_id.dimensionality_reduction_wrapper import DimensionalityReducerTransformer
from FIT_python.pipeline_individual_id.distance_metrics import compute_distances

# Neue Imports für Outlier-Cleaning und Scaling
from FIT_python.pipeline_individual_id.outlier_wrapper import OutlierCleanerTransformer
from FIT_python.pipeline_individual_id.feature_scaler_wrapper import FeatureScalerTransformer

from typing import List, Dict, Union, Optional, Tuple


def run_all_pairwise_projections_parallel(
    comparisons: List[Dict],
    df: pd.DataFrame,
    feature_cols: List[str],
    k_features: Union[int, List[int]] = 15,
    reducers: List[str] = ["lda"],
    selection_method: str = "forward",
    n_components: Union[int, List[int]] = 2,
    outlier_methods: Union[str, List[str], None] = None,
    scaler_methods: Union[str, List[str], None] = None,
    use_sexmodel_prediction: bool = False,
    sexmodel_path: str = None,
    debug: bool = False,
    n_jobs: int = -1
) -> List[Dict]:
    """
    Für jede Paarung:
      0) Falls use_sexmodel_prediction, Sex-Modell laden & predict_proba vorberechnen
      1) Basis-DF bereinigen
      2) Pipeline-Schritte: Outlier-Cleaning & Feature-Scaling
      3) Feature-Selection (einmal mit k_max)
      4) RCV-Set als Komplement
      5) Sex-probas extrahieren & mitteln
      6) Für jede Kombi (outlier, scaler, reducer, n_components, k):
         - apply LDA/PCA/UMAP
         - Abstände berechnen
         - Result-Dict inkl. avg_proba_A/B/R 0/1
    """

    # --- 0) Sex-Modell laden, falls gewünscht ---
    if use_sexmodel_prediction:
        # wenn expliziter Pfad übergeben, ganz normal laden
        if sexmodel_path:
            sex_clf = load(sexmodel_path)
        else:
            # Roh-Ergebnisse einlesen
            raw = pd.read_csv(Path(RESULTS_DATA_DIR) / "raw_results.csv")
            # Zeile mit maximaler test_balanced_accuracy
            best = raw.loc[raw["test_balanced_accuracy"].idxmax()]
            # daraus Dateinamen rekonstruieren: <species>.joblib
            sexmodel_path = Path(RESULTS_DATA_DIR) / "models" / f"{best['species']}.joblib"
            sex_clf = load(sexmodel_path)

    # --- 1) Basis-DF vorbereiten ---
    df2 = df.copy()
    df2[feature_cols] = df2[feature_cols].apply(pd.to_numeric, errors="coerce")
    df_base = df2.reset_index(drop=True)

    # --- 2) predict_proba komplett vorberechnen ---
    if use_sexmodel_prediction:
        proba_all = sex_clf.predict_proba(df_base[feature_cols])
    else:
        proba_all = None

    # --- 3) Parameter-Listen aufbauen ---
    ks    = k_features if isinstance(k_features, (list, tuple)) else [k_features]
    k_max = max(ks)
    ncs   = n_components if isinstance(n_components, (list, tuple)) else [n_components]

    # Outlier-Methoden-Liste
    if isinstance(outlier_methods, (list, tuple)):
        outs = outlier_methods
    elif outlier_methods is not None:
        outs = [outlier_methods]
    else:
        outs = [None]

    # Scaler-Methoden-Liste
    if isinstance(scaler_methods, (list, tuple)):
        scalers = scaler_methods
    elif scaler_methods is not None:
        scalers = [scaler_methods]
    else:
        scalers = [None]

    def process_pair(i: int, comp: Dict) -> List[Dict]:
        out = []
        try:
            ind_a, ind_b   = comp["ind_a"], comp["ind_b"]
            idx_a, idx_b   = comp["samples_a"], comp["samples_b"]
            size_a, size_b = len(idx_a), len(idx_b)

            # Feature-Matrizen A & B
            df_a = df_base.loc[idx_a, feature_cols]
            df_b = df_base.loc[idx_b, feature_cols]

            # RCV-Set als Komplement
            all_idx = np.arange(len(df_base))
            rcv_idx = list(set(all_idx) - set(idx_a) - set(idx_b))
            df_r = df_base.loc[rcv_idx, feature_cols]

            # Labels für Selection
            y_ab = np.concatenate([np.zeros(size_a, int), np.ones(size_b, int)])

            # Sex-probas extrahieren, falls benötigt
            if use_sexmodel_prediction:
                pa = proba_all[idx_a]
                pb = proba_all[idx_b]
                pr = proba_all[rcv_idx]
                avg_A_0, avg_A_1 = float(pa[:,0].mean()), float(pa[:,1].mean())
                avg_B_0, avg_B_1 = float(pb[:,0].mean()), float(pb[:,1].mean())
                avg_R_0, avg_R_1 = float(pr[:,0].mean()), float(pr[:,1].mean())

            # --- 4) Schleife über Outlier-Methoden ---
            for out_method in outs:
                # Outlier-Cleaner initialisieren & fit auf A+B
                if out_method:
                    oc = OutlierCleanerTransformer(method=out_method)
                    X_ab_raw = pd.concat([df_a, df_b], ignore_index=True)
                    oc.fit(X_ab_raw)
                    df_a_proc = oc.transform(df_a)
                    df_b_proc = oc.transform(df_b)
                    df_r_proc = oc.transform(df_r)
                else:
                    df_a_proc = df_a.copy()
                    df_b_proc = df_b.copy()
                    df_r_proc = df_r.copy()

                # --- 5) Schleife über Scaler-Methoden ---
                for scaler_method in scalers:
                    if scaler_method:
                        sc = FeatureScalerTransformer(method=scaler_method)
                        X_ab_in = pd.concat([df_a_proc, df_b_proc], ignore_index=True)
                        sc.fit(X_ab_in)
                        df_a_fs = sc.transform(df_a_proc)
                        df_b_fs = sc.transform(df_b_proc)
                        df_r_fs = sc.transform(df_r_proc)
                    else:
                        df_a_fs = df_a_proc.copy()
                        df_b_fs = df_b_proc.copy()
                        df_r_fs = df_r_proc.copy()

                    # --- 6) Feature-Selection auf A+B ---
                    X_ab_fs = pd.concat([df_a_fs, df_b_fs], ignore_index=True)
                    selector = FeatureSelectionTransformer(method=selection_method, k=k_max)
                    selector.fit(X_ab_fs, y_ab)
                    full_ranking = selector.feature_ranking_

                    # --- 7) Schleifen über Reducer, n_components & k_features ---
                    for reducer in tqdm(reducers, desc=f"[Pair {i}] Reducer", leave=False):
                        supervised = reducer in ("lda", "umap")
                        for nc in tqdm(ncs, desc=f"[Pair {i} / {reducer}] n_comp", leave=False):
                            for k in tqdm(ks, desc=f"[Pair {i} / {reducer} / nc={nc}] k", leave=False):
                                sel_feats = [feat for feat,_ in full_ranking[:k]]

                                da = df_a_fs[sel_feats].copy()
                                db = df_b_fs[sel_feats].copy()
                                dr = df_r_fs[sel_feats].copy()

                                # Sex-probas als Features anhängen
                                if use_sexmodel_prediction:
                                    da["proba_0"], da["proba_1"] = pa[:,0], pa[:,1]
                                    db["proba_0"], db["proba_1"] = pb[:,0], pb[:,1]
                                    dr["proba_0"], dr["proba_1"] = pr[:,0], pr[:,1]

                                # Kombinieren + Labels
                                arr = pd.concat([da, db, dr], ignore_index=True)
                                y_all = np.concatenate([
                                    np.zeros(len(da), int),
                                    np.ones(len(db), int),
                                    np.full(len(dr), 2, int)
                                ])

                                # Clamp für LDA
                                nc_eff = nc
                                if reducer == "lda":
                                    n_cls = len(np.unique(y_all))
                                    nc_eff = min(nc, arr.shape[1], n_cls - 1)
                                    if nc_eff < 1:
                                        if debug:
                                            print(f"[DEBUG] skip LDA pair {i}, k={k}, nc={nc}")
                                        continue

                                # Fit & Transform
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

                                # Result-Dict
                                pipeline_name = (
                                    f"{selection_method}"
                                    f"_{out_method or 'no_out'}"
                                    f"_{scaler_method or 'no_scale'}"
                                    f"_k{k}_{reducer}_nc{nc_eff}"
                                    f"_{'sex_on' if use_sexmodel_prediction else 'sex_off'}"
                                )
                                res = {
                                    "trail_a_id":        comp["trail_a_id"],
                                    "trail_b_id":        comp["trail_b_id"],
                                    "samples_a":         idx_a,
                                    "samples_b":         idx_b,
                                    "ind_a":             ind_a,
                                    "ind_b":             ind_b,
                                    "same_individual":   comp["same_individual"],
                                    "fold":              comp["fold"],
                                    "pipeline":          pipeline_name,
                                    "selection_method":  selection_method,
                                    "reducer":           reducer,
                                    "outlier_method":    out_method,
                                    "scaler_method":     scaler_method,
                                    "k_features":        k,
                                    "n_components":      nc_eff,
                                    "comparison_id":     i,
                                    "min_observations":  min(size_a, size_b),
                                    "max_observations":  max(size_a, size_b),
                                    "use_sexmodel_prediction": use_sexmodel_prediction,
                                    **({
                                        "avg_proba_A_0": avg_A_0,
                                        "avg_proba_A_1": avg_A_1,
                                        "avg_proba_B_0": avg_B_0,
                                        "avg_proba_B_1": avg_B_1,
                                        "avg_proba_R_0": avg_R_0,
                                        "avg_proba_R_1": avg_R_1,
                                    } if use_sexmodel_prediction else {}),
                                    "center_a_x": float(cA[0]) if cA.size>0 else None,
                                    "center_a_y": float(cA[1]) if cA.size>1 else None,
                                    "center_b_x": float(cB[0]) if cB.size>0 else None,
                                    "center_b_y": float(cB[1]) if cB.size>1 else None,
                                    "center_r_x": float(cR[0]) if cR.size>0 else None,
                                    "center_r_y": float(cR[1]) if cR.size>1 else None,
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

    # --- 11) Parallel-Ausführung ---
    with tqdm_joblib(tqdm(desc="Processing Pairs", total=len(comparisons))):
        nested = Parallel(n_jobs=n_jobs)(
            delayed(process_pair)(i, comp)
            for i, comp in enumerate(comparisons)
        )

    # Flatten und zurückgeben
    return [row for group in nested for row in group]
