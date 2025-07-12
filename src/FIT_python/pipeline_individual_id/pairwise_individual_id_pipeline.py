# src/FIT_python/pipeline_individual_id/geometric_pairwise_projection.py

# Standard library
import os
import random
from collections import defaultdict
from itertools import combinations
from pathlib import Path


# Third-party
import numpy as np
import pandas as pd
from joblib import Parallel, delayed, load
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from tqdm import tqdm
from tqdm_joblib import tqdm_joblib

from FIT_python.config import RESULTS_DATA_DIR
from FIT_python.soft_config import SOFT_CONFIG

# Local modules
from FIT_python.pipeline_individual_id.rcv_sampling import generate_rcv
from FIT_python.general_pipeline_steps.feature_selection_wrapper import (
    FeatureSelectionTransformer,
)
from FIT_python.general_pipeline_steps.dimensionality_reduction_wrapper import (
    DimensionalityReducerTransformer,
)
from FIT_python.pipeline_individual_id.distance_metrics import compute_distances
from FIT_python.general_pipeline_steps.outlier_wrapper import OutlierCleanerTransformer
from FIT_python.general_pipeline_steps.feature_scaler_wrapper import (
    FeatureScalerTransformer,
)

from typing import List, Dict, Union, Optional, Tuple
from scipy.spatial.distance import cdist


def run_all_pairwise_projections_parallel(
    comparisons: List[Dict],
    df: pd.DataFrame,
    feature_cols: List[str],
    k_features: Union[int, List[int]] = SOFT_CONFIG["pipeline_individual_id"][
        "pairwise_defaults"
    ]["k_features"],
    reducers: List[str] = SOFT_CONFIG["pipeline_individual_id"]["pairwise_defaults"][
        "reducers"
    ],
    selection_method: str = SOFT_CONFIG["pipeline_individual_id"]["pairwise_defaults"][
        "selection_method"
    ],
    n_components: Union[int, List[int]] = SOFT_CONFIG["pipeline_individual_id"][
        "pairwise_defaults"
    ]["n_components"],
    outlier_methods: Union[str, List[str], None] = None,
    scaler_methods: Union[str, List[str], None] = None,
    use_sexmodel_prediction: bool = False,
    sexmodel_path: str = None,
    debug: bool = False,
    n_jobs: int = -1,
) -> List[Dict]:
    """Process all pairwise projections.

    Steps
    -----
    0. If ``use_sexmodel_prediction`` is ``True`` load the sex model and
       precompute ``predict_proba``. ``sexmodel_path`` must point to a valid ``.joblib`` file.
    1. Clean the base DataFrame.
    2. Apply pipeline steps: outlier cleaning and feature scaling.
    3. Perform feature selection once with ``k_max`` on the **scaled** data.
    4. Use an RCV set as the complement.
    5. Extract and average sex probabilities.
    6. For each combination of ``outlier``, ``scaler``, ``reducer``, ``n_components`` and ``k``:
       - apply the dimensionality reduction
       - compute distances
       - record results including average probabilities for A/B/R
    """

    # --- 0) load sex model if requested ---
    if use_sexmodel_prediction:
        if not sexmodel_path:
            raise ValueError(
                "sexmodel_path must be provided when use_sexmodel_prediction=True"
            )

        model_fp = Path(sexmodel_path)
        if not model_fp.is_file():
            raise FileNotFoundError(f"Sex model file not found: {sexmodel_path}")

        sex_clf = load(model_fp)

    # --- 1) prepare base DataFrame ---
    df2 = df.copy()
    df2["id"] = df2["id"].astype(str)
    df2[feature_cols] = df2[feature_cols].apply(pd.to_numeric, errors="coerce")
    df_base = df2.reset_index(drop=True)

    # mapping from original DataFrame index to position after reset
    index_map = {str(idx): pos for pos, idx in enumerate(df2.index)}

    # --- 2) pre-compute ``predict_proba`` for all samples ---
    if use_sexmodel_prediction:
        proba_all = pd.DataFrame(
            sex_clf.predict_proba(df_base[feature_cols]),
            index=df_base.index,
        )
    else:
        proba_all = None

    # --- 3) build parameter lists ---
    ks = k_features if isinstance(k_features, (list, tuple)) else [k_features]
    k_max = max(ks)
    ncs = n_components if isinstance(n_components, (list, tuple)) else [n_components]

    if isinstance(outlier_methods, (list, tuple)):
        outs = outlier_methods
    elif outlier_methods is not None:
        outs = [outlier_methods]
    else:
        outs = [None]

    if isinstance(scaler_methods, (list, tuple)):
        scalers = scaler_methods
    elif scaler_methods is not None:
        scalers = [scaler_methods]
    else:
        scalers = [None]

    def process_pair(i: int, comp: Dict) -> List[Dict]:
        out = []
        ind_a, ind_b = comp["ind_a"], comp["ind_b"]
        ids_a = [str(i) for i in comp["samples_a"]]
        ids_b = [str(i) for i in comp["samples_b"]]
        idx_a = [index_map[i] for i in ids_a]
        idx_b = [index_map[i] for i in ids_b]

        size_a, size_b = len(idx_a), len(idx_b)

        trail_a_id = comp["trail_a_id"]
        trail_b_id = comp["trail_b_id"]

        # feature matrices A & B
        df_a = df_base.iloc[idx_a][feature_cols]
        df_b = df_base.iloc[idx_b][feature_cols]

        # RCV set as the complement
        all_idx = set(range(len(df_base)))
        rcv_idx = list(all_idx - set(idx_a) - set(idx_b))
        df_r = df_base.iloc[rcv_idx][feature_cols]

        # labels for feature selection
        y_ab = np.concatenate([np.zeros(size_a, int), np.ones(size_b, int)])

        # extract sex probabilities if required
        if use_sexmodel_prediction:
            pa = proba_all.iloc[idx_a].to_numpy()
            pb = proba_all.iloc[idx_b].to_numpy()
            pr = proba_all.iloc[rcv_idx].to_numpy()
            avg_A_0, avg_A_1 = float(pa[:, 0].mean()), float(pa[:, 1].mean())
            avg_B_0, avg_B_1 = float(pb[:, 0].mean()), float(pb[:, 1].mean())
            avg_R_0, avg_R_1 = float(pr[:, 0].mean()), float(pr[:, 1].mean())

        # 4) iterate over outlier and scaler methods
        for out_method in outs:
            for scaler_method in scalers:
                steps = []
                if out_method:
                    steps.append(
                        ("outlier", OutlierCleanerTransformer(method=out_method))
                    )
                if scaler_method:
                    steps.append(
                        ("scale", FeatureScalerTransformer(method=scaler_method))
                    )
                selector = FeatureSelectionTransformer(method=selection_method, k=k_max)
                steps.append(("select", selector))
                pipe = Pipeline(steps)

                X_ab_raw = pd.concat([df_a, df_b], ignore_index=True)
                pipe.fit(X_ab_raw, y_ab)
                full_ranking = selector.feature_ranking_

                df_a_fs = pipe.transform(df_a)
                df_b_fs = pipe.transform(df_b)
                df_r_fs = pipe.transform(df_r)

                # ``Pipeline.transform`` may return an ``np.ndarray`` depending
                # on the scikit-learn version. The following steps expect a
                # ``DataFrame`` with column names, therefore convert the array
                # back to a ``DataFrame`` if necessary.
                if not isinstance(df_a_fs, pd.DataFrame):
                    feat_names = selector.get_feature_names_out()
                    df_a_fs = pd.DataFrame(
                        df_a_fs, columns=feat_names, index=df_a.index
                    )
                    df_b_fs = pd.DataFrame(
                        df_b_fs, columns=feat_names, index=df_b.index
                    )
                    df_r_fs = pd.DataFrame(
                        df_r_fs, columns=feat_names, index=df_r.index
                    )

                # 5) iterate over reducers, n_components and k_features
                for reducer in tqdm(reducers, desc=f"[Pair {i}] Reducer", leave=False):
                    supervised = reducer in ("lda", "umap")
                    for nc in tqdm(
                        ncs, desc=f"[Pair {i} / {reducer}] n_comp", leave=False
                    ):
                        for k in tqdm(
                            ks, desc=f"[Pair {i} / {reducer} / nc={nc}] k", leave=False
                        ):
                            sel_feats = [feat for feat, _ in full_ranking[:k]]

                            da = df_a_fs[sel_feats].copy()
                            db = df_b_fs[sel_feats].copy()
                            dr = df_r_fs[sel_feats].copy()

                            # append sex probabilities as features
                            if use_sexmodel_prediction:
                                da["proba_0"], da["proba_1"] = pa[:, 0], pa[:, 1]
                                db["proba_0"], db["proba_1"] = pb[:, 0], pb[:, 1]
                                dr["proba_0"], dr["proba_1"] = pr[:, 0], pr[:, 1]

                            # combine and create labels
                            arr = pd.concat([da, db, dr], ignore_index=True)
                            y_all = np.concatenate(
                                [
                                    np.zeros(len(da), int),
                                    np.ones(len(db), int),
                                    np.full(len(dr), 2, int),
                                ]
                            )

                            # clamp for LDA
                            nc_eff = nc
                            if reducer == "lda":
                                n_cls = len(np.unique(y_all))
                                nc_eff = min(nc, arr.shape[1], n_cls - 1)
                                if nc_eff < 1:
                                    if debug:
                                        print(
                                            f"[DEBUG] skip LDA pair {i}, k={k}, nc={nc}"
                                        )
                                    continue

                            # Fit & Transform
                            dr_model = DimensionalityReducerTransformer(
                                method=reducer,
                                n_components=nc_eff,
                                supervised=supervised,
                            )
                            dr_model.fit(arr, y_all if supervised else None)
                            coords = dr_model.transform(arr)

                            ca = coords[: len(da)]
                            cb = coords[len(da) : len(da) + len(db)]
                            cr = coords[len(da) + len(db) :]

                            cA, cB, cR = (
                                ca.mean(axis=0),
                                cb.mean(axis=0),
                                cr.mean(axis=0),
                            )
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
                                "trail_a_id": comp["trail_a_id"],
                                "trail_b_id": comp["trail_b_id"],
                                "samples_a": ids_a,
                                "samples_b": ids_b,
                                "ind_a": ind_a,
                                "ind_b": ind_b,
                                # store as string to avoid mixed bool/object dtype
                                "same_individual": str(comp["same_individual"]),
                                "fold": comp["fold"],
                                "pipeline": pipeline_name,
                                "selection_method": selection_method,
                                "reducer": reducer,
                                "outlier_method": out_method,
                                "scaler_method": scaler_method,
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
                                "coords_a_y": (
                                    ca[:, 1].tolist() if ca.shape[1] > 1 else []
                                ),
                                "coords_b_x": cb[:, 0].tolist(),
                                "coords_b_y": (
                                    cb[:, 1].tolist() if cb.shape[1] > 1 else []
                                ),
                                "coords_r_x": cr[:, 0].tolist(),
                                "coords_r_y": (
                                    cr[:, 1].tolist() if cr.shape[1] > 1 else []
                                ),
                                "selected_features": sel_feats,
                                "selected_scores": [
                                    score for _, score in full_ranking[:k]
                                ],
                            }
                            for m, v in dists.items():
                                res[f"dist_{m}"] = float(v)

                                # if distance could not be computed (e.g. Mahalanobis without covariance)
                                if np.isnan(v):
                                    res[f"mean_{m}_between"] = None
                                    res[f"median_{m}_between"] = None
                                    res[f"mean_{m}_within_a"] = None
                                    res[f"median_{m}_within_a"] = None
                                    res[f"mean_{m}_within_b"] = None
                                    res[f"median_{m}_within_b"] = None
                                    continue

                                # additional metrics: pairwise distances between A and B
                                A = da.to_numpy()
                                B = db.to_numpy()
                                metric_name = "cityblock" if m == "manhattan" else m
                                d_ab = cdist(A, B, metric=metric_name)
                                flat_ab = d_ab.ravel()
                                res[f"mean_{m}_between"] = float(np.mean(flat_ab))
                                res[f"median_{m}_between"] = float(np.median(flat_ab))

                                # within A
                                if len(A) > 1:
                                    d_aa = cdist(A, A, metric=metric_name)
                                    iu = np.triu_indices(len(A), k=1)
                                    flat_aa = d_aa[iu]
                                    res[f"mean_{m}_within_a"] = float(np.mean(flat_aa))
                                    res[f"median_{m}_within_a"] = float(
                                        np.median(flat_aa)
                                    )
                                else:
                                    res[f"mean_{m}_within_a"] = None
                                    res[f"median_{m}_within_a"] = None

                                # within B
                                if len(B) > 1:
                                    d_bb = cdist(B, B, metric=metric_name)
                                    iu = np.triu_indices(len(B), k=1)
                                    flat_bb = d_bb[iu]
                                    res[f"mean_{m}_within_b"] = float(np.mean(flat_bb))
                                    res[f"median_{m}_within_b"] = float(
                                        np.median(flat_bb)
                                    )
                                else:
                                    res[f"mean_{m}_within_b"] = None
                                    res[f"median_{m}_within_b"] = None

                            out.append(res)

        return out

    # --- 6) parallel execution ---
    with tqdm_joblib(tqdm(desc="Processing Pairs", total=len(comparisons))):
        nested = Parallel(n_jobs=n_jobs)(
            delayed(process_pair)(i, comp) for i, comp in enumerate(comparisons)
        )

    # flatten nested list and return
    return [row for group in nested for row in group]


def run_embedding_once_pipeline(
    comparisons: List[Dict],
    df: pd.DataFrame,
    *,
    feature_cols: List[str],
    k_features: int = SOFT_CONFIG["pipeline_individual_id"]["pairwise_defaults"][
        "k_features"
    ],
    reducer: str = SOFT_CONFIG["pipeline_individual_id"]["pairwise_defaults"][
        "reducers"
    ][0],
    selection_method: str = SOFT_CONFIG["pipeline_individual_id"]["pairwise_defaults"][
        "selection_method"
    ],
    n_components: int = SOFT_CONFIG["pipeline_individual_id"]["pairwise_defaults"][
        "n_components"
    ],
    outlier_method: str | None = None,
    scaler_method: str | None = None,
    use_sexmodel_prediction: bool = False,
    sexmodel_path: str | None = None,
    debug: bool = False,
) -> List[Dict]:
    """Convenience wrapper that processes comparisons sequentially."""

    return run_all_pairwise_projections_parallel(
        comparisons,
        df,
        feature_cols=feature_cols,
        k_features=k_features,
        reducers=[reducer],
        selection_method=selection_method,
        n_components=n_components,
        outlier_methods=outlier_method,
        scaler_methods=scaler_method,
        use_sexmodel_prediction=use_sexmodel_prediction,
        sexmodel_path=sexmodel_path,
        debug=debug,
        n_jobs=1,
    )
