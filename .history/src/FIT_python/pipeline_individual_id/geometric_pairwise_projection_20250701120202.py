import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from typing import List, Dict
from collections import defaultdict
import random
from FIT_python.pipeline_individual_id.rcv_sampling import generate_rcv
from FIT_python.pipeline_individual_id.feature_selection_wrapper import FeatureSelectionTransformer
from collections import defaultdict
import random


def geometric_pairwise_projection(
    comparison: Dict,
    df: pd.DataFrame,
    rcv_df: pd.DataFrame,
    feature_cols: List[str],
    k_features: int = 10,
    reducer: str = "pca",
    n_components: int = 2,
    debug: bool = False
):
    idx_a = comparison["samples_a"]
    idx_b = comparison["samples_b"]
    df_a = df.loc[idx_a, feature_cols].copy()
    df_b = df.loc[idx_b, feature_cols].copy()
    df_r = rcv_df[feature_cols].copy()

    X_ab = pd.concat([df_a, df_b], ignore_index=True)
    y_ab = np.concatenate([
        np.zeros(len(df_a), dtype=int),
        np.ones(len(df_b), dtype=int)
    ])

    selector = FeatureSelectionTransformer(method="forward", k=k_features)
    selector.fit(X_ab, y_ab)
    selected_features = selector.selected_features_

    if debug:
        print("Selected features:", selected_features)

    if reducer == "pca":
        reducer_model = PCA(n_components=n_components)
    elif reducer == "lda":
        reducer_model = LinearDiscriminantAnalysis(n_components=1)
    else:
        raise ValueError("Reducer muss 'pca' oder 'lda' sein")

    reducer_model.fit(X_ab[selected_features], y_ab)
    coords_a = reducer_model.transform(df_a[selected_features])
    coords_b = reducer_model.transform(df_b[selected_features])
    coords_r = reducer_model.transform(df_r[selected_features])

    center_a = coords_a.mean(axis=0)
    center_b = coords_b.mean(axis=0)
    center_r = coords_r.mean(axis=0)

    result = {
        "ind_a": comparison["ind_a"],
        "ind_b": comparison["ind_b"],
        "selected_features": selected_features,
        "center_distance_ab": np.linalg.norm(center_a - center_b),
        "center_distance_rcv_a": np.linalg.norm(center_r - center_a),
        "center_distance_rcv_b": np.linalg.norm(center_r - center_b),
        "coords": {
            "A": coords_a.tolist(),
            "B": coords_b.tolist(),
            "RCV": coords_r.tolist(),
        }
    }
    return result




from collections import defaultdict
import random
from typing import List, Dict
import pandas as pd

def generate_pairwise_comparisons_from_df(
    df: pd.DataFrame,
    id_col: str = "individual_id",
    group_sizes: List[int] = [3, 5, 7, 10],
    n_repeats: int = 5,
    mode: str = 'both',
    selfmatch_factor: float = 2.0
) -> List[Dict]:
    """
    Erzeuge Paarvergleiche zwischen Trails (verschiedene oder gleiche Individuen).

    Parameter
    ---------
    df : pd.DataFrame
        Eingabedaten mit mindestens einer 'individual_id'-Spalte.
    id_col : str
        Spaltenname der individuellen ID.
    group_sizes : list of int
        Gruppengrößen (Anzahl Trails), die für die Vergleiche verwendet werden sollen.
    n_repeats : int
        Wie oft jeder Vergleichstyp wiederholt werden soll.
    mode : str
        "symmetric", "asymmetric" oder "both", je nach erlaubten Größenkombinationen.
    selfmatch_factor : float
        Multiplikator für die Anzahl der Selbstvergleiche.

    Rückgabe
    --------
    List[Dict]
        Liste von Vergleichsdictionaries mit Trails und Metadaten.
    """
    individuals = defaultdict(list)
    for idx, row in df.iterrows():
        individuals[row[id_col]].append(idx)

    comparisons = []

    for size_a in group_sizes:
        for size_b in group_sizes:
            if mode == 'symmetric' and size_a != size_b:
                continue
            if mode == 'asymmetric' and size_a == size_b:
                continue

            eligible_inds_a = [ind for ind, samples in individuals.items() if len(samples) >= size_a]
            eligible_inds_b = [ind for ind, samples in individuals.items() if len(samples) >= size_b]

            # Cross-individual comparisons
            for ind_a in eligible_inds_a:
                for ind_b in eligible_inds_b:
                    if ind_a >= ind_b:
                        continue
                    for _ in range(n_repeats):
                        samples_a = random.sample(individuals[ind_a], size_a)
                        samples_b = random.sample(individuals[ind_b], size_b)
                        comparisons.append({
                            'ind_a': ind_a,
                            'ind_b': ind_b,
                            'size_a': size_a,
                            'size_b': size_b,
                            'samples_a': samples_a,
                            'samples_b': samples_b,
                            'same_individual': False
                        })

            # Same-individual comparisons
            for ind in individuals:
                if len(individuals[ind]) < size_a + size_b:
                    continue
                self_repeats = int(n_repeats * selfmatch_factor)
                for _ in range(self_repeats):
                    combined = random.sample(individuals[ind], size_a + size_b)
                    samples_a = combined[:size_a]
                    samples_b = combined[size_a:]
                    comparisons.append({
                        'ind_a': ind,
                        'ind_b': ind,
                        'size_a': size_a,
                        'size_b': size_b,
                        'samples_a': samples_a,
                        'samples_b': samples_b,
                        'same_individual': True
                    })

    return comparisons


def run_all_pairwise_projections(
    comparisons: List[Dict],
    df: pd.DataFrame,
    feature_cols: List[str],
    k_features: int = 10,
    reducer: str = "pca",
    selection_method: str = "forward",
    n_components: int = 2,
    debug: bool = False
):
    results = []
    df = df.copy()
    df_features = df[feature_cols].apply(pd.to_numeric, errors="coerce")
    df.update(df_features)
    df_base = df.reset_index(drop=True)

    for i, comp in enumerate(comparisons):
        if debug and i % 100 == 0:
            print(f"Processing comparison {i+1}/{len(comparisons)}")

        try:
            idx_a = comp["samples_a"]
            idx_b = comp["samples_b"]
            df_a = df_base.loc[idx_a, feature_cols]
            df_b = df_base.loc[idx_b, feature_cols]
            X_ab = pd.concat([df_a, df_b], ignore_index=True)
            y_ab = np.concatenate([
                np.zeros(len(df_a), dtype=int),
                np.ones(len(df_b), dtype=int)
            ])

            selector = FeatureSelectionTransformer(method=selection_method, k=k_features)
            selector.fit(X_ab, y_ab)
            selected_features = selector.selected_features_

            exclude = idx_a + idx_b
            df_r = generate_rcv(df_base, exclude)[selected_features]

            df_all = pd.concat([
                df_a[selected_features],
                df_b[selected_features],
                df_r
            ], ignore_index=True)
            y_all = np.concatenate([
                np.zeros(len(df_a), dtype=int),
                np.ones(len(df_b), dtype=int),
                np.full(len(df_r), 2, dtype=int)
            ])

            if reducer == "pca":
                reducer_model = PCA(n_components=n_components)
            elif reducer == "lda":
                reducer_model = LinearDiscriminantAnalysis(n_components=min(n_components, len(np.unique(y_all)) - 1))
            else:
                raise ValueError("Reducer muss 'pca' oder 'lda' sein")

            reducer_model.fit(df_all, y_all)
            coords_all = reducer_model.transform(df_all)

            coords_a = coords_all[:len(df_a)]
            coords_b = coords_all[len(df_a):len(df_a)+len(df_b)]
            coords_r = coords_all[len(df_a)+len(df_b):]

            center_a = coords_a.mean(axis=0)
            center_b = coords_b.mean(axis=0)
            center_r = coords_r.mean(axis=0)

            result = {
                "ind_a": comp["ind_a"],
                "ind_b": comp["ind_b"],
                "same_individual": comp["same_individual"],
                "selected_features": selected_features,
                "selection_method": selection_method,
                "reducer": reducer,
                "k_features": k_features,
                "group_size_a": len(idx_a),
                "group_size_b": len(idx_b),
                "center_distance_ab": np.linalg.norm(center_a - center_b),
                "center_distance_rcv_a": np.linalg.norm(center_a - center_r),
                "center_distance_rcv_b": np.linalg.norm(center_b - center_r),
                "comparison_id": i,
            }

            results.append(result)

        except Exception as e:
            if debug:
                print(f"[ERROR] Comparison {i} failed: {e}")

    return results

from joblib import Parallel, delayed
from tqdm import tqdm

ddef run_all_pairwise_projections_parallel(
    comparisons: List[Dict],
    df: pd.DataFrame,
    feature_cols: List[str],
    k_features: Union[int, List[int]] = 15,
    reducers: List[str] = ["lda"],
    selection_method: str = "forward",
    n_components: int = 2,
    debug: bool = False,
    n_jobs: int = -1
):
    import pandas as pd
    import numpy as np
    from joblib import Parallel, delayed
    from tqdm import tqdm
    from FIT_python.pipeline_individual_id.feature_selection_wrapper import FeatureSelectionTransformer
    from FIT_python.pipeline_individual_id.dimensionality_reduction_wrapper import DimensionalityReducerTransformer
    from FIT_python.pipeline_individual_id.distance_measures import compute_distances
    from FIT_python.pipeline_individual_id.rcv_sampling import generate_rcv

    # Ensure k_features is a list
    ks = k_features if isinstance(k_features, (list, tuple)) else [k_features]
    k_max = max(ks)

    df = df.copy()
    df_features = df[feature_cols].apply(pd.to_numeric, errors="coerce")
    df.update(df_features)
    df_base = df.reset_index(drop=True)

    all_results = []

    # STEP 1: Feature Selection ONCE per pair using k_max
    for i, comp in tqdm(list(enumerate(comparisons)), desc="Feature Selection & Projection"):
        try:
            idx_a = comp["samples_a"]
            idx_b = comp["samples_b"]
            df_a = df_base.loc[idx_a, feature_cols]
            df_b = df_base.loc[idx_b, feature_cols]
            X_ab = pd.concat([df_a, df_b], ignore_index=True)
            y_ab = np.concatenate([
                np.zeros(len(df_a), dtype=int),
                np.ones(len(df_b), dtype=int)
            ])

            # run feature selection once
            selector = FeatureSelectionTransformer(method=selection_method, k=k_max)
            selector.fit(X_ab, y_ab)
            full_ranking = selector.feature_ranking_

            for k in ks:
                selected_features = [feat for feat,_ in full_ranking[:k]]
                for reducer in reducers:
                    # Prepare data for dim red
                    subset_a = df_a[selected_features]
                    subset_b = df_b[selected_features]
                    exclude = idx_a + idx_b
                    subset_r = generate_rcv(df_base, exclude)[selected_features]
                    df_all = pd.concat([subset_a, subset_b, subset_r], ignore_index=True)
                    y_all = np.concatenate([
                        np.zeros(len(subset_a), dtype=int),
                        np.ones(len(subset_b), dtype=int),
                        np.full(len(subset_r), 2, dtype=int)
                    ])

                    supervised = reducer in ("lda", "umap")
                    dr = DimensionalityReducerTransformer(
                        method=reducer,
                        n_components=n_components,
                        supervised=supervised
                    )
                    dr.fit(df_all, y_all if supervised else None)
                    coords_all = dr.transform(df_all)

                    coords_a = coords_all[:len(subset_a)]
                    coords_b = coords_all[len(subset_a):len(subset_a)+len(subset_b)]
                    coords_r = coords_all[len(subset_a)+len(subset_b):]

                    center_a = coords_a.mean(axis=0)
                    center_b = coords_b.mean(axis=0)
                    center_r = coords_r.mean(axis=0)

                    distances = compute_distances(center_a, center_b)

                    result = {
                        "ind_a": comp["ind_a"],
                        "ind_b": comp["ind_b"],
                        "same_individual": comp["same_individual"],
                        "selection_method": selection_method,
                        "reducer": reducer,
                        "k_features": k,
                        "comparison_id": i,

                        "center_a_x": center_a[0] if len(center_a)>0 else None,
                        "center_a_y": center_a[1] if len(center_a)>1 else None,
                        "center_b_x": center_b[0] if len(center_b)>0 else None,
                        "center_b_y": center_b[1] if len(center_b)>1 else None,
                        "center_r_x": center_r[0] if len(center_r)>0 else None,
                        "center_r_y": center_r[1] if len(center_r)>1 else None,

                                                # coordinates as separate x/y lists
                        "coords_a_x": coords_a[:,0].tolist() if coords_a.shape[1]>0 else [],
                        "coords_a_y": coords_a[:,1].tolist() if coords_a.shape[1]>1 else [],
                        "coords_b_x": coords_b[:,0].tolist() if coords_b.shape[1]>0 else [],
                        "coords_b_y": coords_b[:,1].tolist() if coords_b.shape[1]>1 else [],
                        "coords_r_x": coords_r[:,0].tolist() if coords_r.shape[1]>0 else [],
                        "coords_r_y": coords_r[:,1].tolist() if coords_r.shape[1]>1 else [],

                        "dist_euclidean": distances["euclidean"],
                        "dist_manhattan": distances["manhattan"],
                        "dist_cosine": distances["cosine"],
                        "dist_chebyshev": distances["chebyshev"],
                        "dist_canberra": distances["canberra"],
                        "dist_braycurtis": distances["braycurtis"],
                        "dist_mahalanobis": distances["mahalanobis"],

                        "min_observations": min(len(idx_a), len(idx_b)),
                        "max_observations": max(len(idx_a), len(idx_b)),
                        "n_selected_features": k,
                    }
                    all_results.append(result)

        except Exception as e:
            if debug:
                print(f"[ERROR] Pair {i} failed: {e}")

    return all_results
