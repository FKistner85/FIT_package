import numpy as np
import pandas as pd
from typing import List, Dict, Iterable
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score

from .outlier_wrapper import OutlierCleanerTransformer
from .feature_scaler_wrapper import FeatureScalerTransformer
from .feature_selection_wrapper import FeatureSelectionTransformer
from .dimensionality_reduction_wrapper import DimensionalityReducerTransformer
from .generate_trails_and_trailpairs import generate_pairwise_comparisons_from_df


def _run_pair_pipeline(pair: Dict, df: pd.DataFrame, feature_cols: Iterable[str], train_indices: Iterable[int], *,
                       selection_method: str = "forward", reducer: str = "lda", n_components: int = 2,
                       k_features: int = 15, outlier_method: str | None = None, scaler_method: str | None = None) -> Dict:
    """Process a single comparison with an RCV complement from the training set."""
    df_base = df.copy()
    df_base[list(feature_cols)] = df_base[list(feature_cols)].apply(pd.to_numeric, errors="coerce")

    idx_a = pair["samples_a"]
    idx_b = pair["samples_b"]
    df_a = df_base.loc[idx_a, feature_cols]
    df_b = df_base.loc[idx_b, feature_cols]

    all_idx = set(df.index)
    train_idx = set(train_indices)
    rcv_idx = list(all_idx - set(idx_a) - set(idx_b) - train_idx)
    df_r = df_base.loc[rcv_idx, feature_cols]

    y_ab = np.concatenate([np.zeros(len(df_a), int), np.ones(len(df_b), int)])
    steps = []
    if outlier_method:
        steps.append(("outlier", OutlierCleanerTransformer(method=outlier_method)))
    if scaler_method:
        steps.append(("scale", FeatureScalerTransformer(method=scaler_method)))
    selector = FeatureSelectionTransformer(method=selection_method, k=k_features)
    steps.append(("select", selector))
    pipe = Pipeline(steps)

    X_ab = pd.concat([df_a, df_b], ignore_index=True)
    pipe.fit(X_ab, y_ab)
    df_a_t = pipe.transform(df_a)
    df_b_t = pipe.transform(df_b)
    df_r_t = pipe.transform(df_r)
    if not isinstance(df_a_t, pd.DataFrame):
        feats = selector.get_feature_names_out()
        df_a_t = pd.DataFrame(df_a_t, columns=feats, index=df_a.index)
        df_b_t = pd.DataFrame(df_b_t, columns=feats, index=df_b.index)
        df_r_t = pd.DataFrame(df_r_t, columns=feats, index=df_r.index)

    arr = pd.concat([df_a_t, df_b_t, df_r_t], ignore_index=True)
    y_all = np.concatenate([
        np.zeros(len(df_a_t), int),
        np.ones(len(df_b_t), int),
        np.full(len(df_r_t), 2, int),
    ])

    supervised = reducer in ("lda", "umap")
    nc_eff = n_components
    if reducer == "lda":
        n_cls = len(np.unique(y_all))
        nc_eff = min(n_components, arr.shape[1], n_cls - 1)
        if nc_eff < 1:
            return {"same_individual": pair["same_individual"], "dist_euclidean": np.nan}
    dr = DimensionalityReducerTransformer(method=reducer, n_components=nc_eff, supervised=supervised)
    dr.fit(arr, y_all if supervised else None)
    coords = dr.transform(arr)

    cA = coords[: len(df_a_t)].mean(axis=0)
    cB = coords[len(df_a_t): len(df_a_t) + len(df_b_t)].mean(axis=0)
    dist = float(np.linalg.norm(cA - cB))
    return {"same_individual": pair["same_individual"], "dist_euclidean": dist}


def _compute_overlap_rate(d_same: List[float], d_diff: List[float]) -> float:
    """Histogram intersection between same and different distance distributions."""
    if not d_same or not d_diff:
        return 0.0
    combined = np.array(d_same + d_diff)
    bins = np.histogram_bin_edges(combined, bins="fd")
    h_same, _ = np.histogram(d_same, bins=bins, density=True)
    h_diff, _ = np.histogram(d_diff, bins=bins, density=True)
    overlap = np.minimum(h_same, h_diff).sum() * np.diff(bins).mean()
    return float(overlap)


def sequential_cv_test_pairs(
    df: pd.DataFrame,
    feature_cols: List[str],
    *,
    n_repeats: int = 5,
    seed: int = 0,
    selection_method: str = "forward",
    reducer: str = "lda",
    n_components: int = 2,
    k_features: int = 15,
    outlier_method: str | None = None,
    scaler_method: str | None = None,
    pair_kwargs: dict | None = None,
) -> pd.DataFrame:
    """Run sequential CV with local RCV and return aggregated metrics."""
    pair_kwargs = pair_kwargs or {}
    defaults = {"n_folds": 1}
    defaults.update(pair_kwargs)
    comps, _ = generate_pairwise_comparisons_from_df(df, **defaults)
    animals = sorted(df["individual_id"].unique())
    all_indices = set(df.index)
    metrics: List[Dict] = []

    for rep in range(n_repeats):
        rng = np.random.default_rng(seed + rep)
        permuted = rng.permutation(animals).tolist()
        for n_test in (3, 6, 9, 12):
            test_animals = set(permuted[:n_test])
            train_animals = set(animals) - test_animals
            train_pairs = [c for c in comps if c["ind_a"] in train_animals and c["ind_b"] in train_animals]
            test_pairs = [c for c in comps if c["ind_a"] in test_animals and c["ind_b"] in test_animals]
            train_indices = df[df["individual_id"].isin(train_animals)].index

            res_train = [_run_pair_pipeline(p, df, feature_cols, train_indices,
                                           selection_method=selection_method,
                                           reducer=reducer, n_components=n_components,
                                           k_features=k_features,
                                           outlier_method=outlier_method,
                                           scaler_method=scaler_method)
                          for p in train_pairs]
            res_test = [_run_pair_pipeline(p, df, feature_cols, train_indices,
                                          selection_method=selection_method,
                                          reducer=reducer, n_components=n_components,
                                          k_features=k_features,
                                          outlier_method=outlier_method,
                                          scaler_method=scaler_method)
                         for p in test_pairs]
            X_train = np.array([[r["dist_euclidean"]] for r in res_train])
            y_train = np.array([bool(r["same_individual"]) for r in res_train], int)
            X_test = np.array([[r["dist_euclidean"]] for r in res_test])
            y_test = np.array([bool(r["same_individual"]) for r in res_test], int)
            if len(np.unique(y_train)) < 2:
                continue
            clf = LogisticRegression().fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            y_score = clf.predict_proba(X_test)[:, 1]
            acc = accuracy_score(y_test, y_pred)
            roc = roc_auc_score(y_test, y_score)
            dist_same = [r["dist_euclidean"] for r in res_test if r["same_individual"]]
            dist_diff = [r["dist_euclidean"] for r in res_test if not r["same_individual"]]
            overlap = _compute_overlap_rate(dist_same, dist_diff)
            metrics.append({"repeat": rep, "test_size": n_test, "accuracy": acc,
                             "roc_auc": roc, "overlap": overlap})

    if not metrics:
        return pd.DataFrame()

    df_metrics = pd.DataFrame(metrics)
    summary = (
        df_metrics.groupby("test_size")[["accuracy", "roc_auc", "overlap"]]
        .agg(["mean", "std"])
        .reset_index()
    )
    return summary
