import pandas as pd
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_selection import SelectKBest, f_classif
from typing import List, Dict
import numpy as np
import os
import pandas as pd
from collections import defaultdict, Counter
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
    # Samplegruppen extrahieren
    idx_a = comparison["samples_a"]
    idx_b = comparison["samples_b"]
    df_a = df.loc[idx_a, feature_cols].copy()
    df_b = df.loc[idx_b, feature_cols].copy()

    # RCV aus anderem Split
    df_r = rcv_df[feature_cols].copy()

    # Labels für supervised Auswahl
    X_ab = pd.concat([df_a, df_b], ignore_index=True)
    y_ab = np.array([0] * len(df_a) + [1] * len(df_b))

    # Feature Selection (A vs B)
    selector = SelectKBest(score_func=f_classif, k=min(k_features, len(feature_cols)))
    X_ab_selected = selector.fit_transform(X_ab, y_ab)
    selected_features = [f for (f, sel) in zip(feature_cols, selector.get_support()) if sel]

    if debug:
        print("Selected features:", selected_features)

    # Dim-Reduktion (z. B. PCA oder LDA)
    if reducer == "pca":
        reducer_model = PCA(n_components=n_components)
    elif reducer == "lda":
        reducer_model = LinearDiscriminantAnalysis(n_components=1)
    else:
        raise ValueError("Reducer muss 'pca' oder 'lda' sein")

    reducer_model.fit(X_ab[selected_features], y_ab)

    # Projektion A, B, RCV
    coords_a = reducer_model.transform(df_a[selected_features])
    coords_b = reducer_model.transform(df_b[selected_features])
    coords_r = reducer_model.transform(df_r[selected_features])

    # Zentren berechnen
    center_a = coords_a.mean(axis=0)
    center_b = coords_b.mean(axis=0)
    center_r = coords_r.mean(axis=0)

    # Distanzen
    dist_ab = np.linalg.norm(center_a - center_b)
    dist_ra = np.linalg.norm(center_r - center_a)
    dist_rb = np.linalg.norm(center_r - center_b)

    # Ausgabe
    result = {
        "ind_a": comparison["ind_a"],
        "ind_b": comparison["ind_b"],
        "selected_features": selected_features,
        "center_distance_ab": dist_ab,
        "center_distance_rcv_a": dist_ra,
        "center_distance_rcv_b": dist_rb,
        "coords": {
            "A": coords_a.tolist(),
            "B": coords_b.tolist(),
            "RCV": coords_r.tolist(),
        }
    }

    return result

def generate_pairwise_comparisons_from_df(
    df: pd.DataFrame,
    id_col: str = "individual_id",
    group_sizes=[3, 5, 7, 10],
    n_repeats=5,
    mode='both',
    selfmatch_factor=2.0
):
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