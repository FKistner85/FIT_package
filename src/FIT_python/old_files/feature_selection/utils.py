# src/FIT_python/feature_utils.py

import pandas as pd
import numpy as np
from scipy.stats import f
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif, chi2, mutual_info_classif
from sklearn.linear_model import LogisticRegression

from FIT_python.config import (
    FS_P_THRESH,
    FS_MIN_NUM_FEATURES,
    FS_N_JOBS,
    FS_DEFAULT_METHODS,
    FS_TARGET_FEATURE_COUNTS
)

def ancova_f_test_with_pinv(X_cov: np.ndarray, group_dummies: np.ndarray, response: np.ndarray):
    """ANCOVA F-test using Moore-Penrose pseudoinverse."""
    n = len(response)
    X_full = np.hstack([np.ones((n, 1)), X_cov, group_dummies])
    H_full = X_full @ np.linalg.pinv(X_full.T @ X_full) @ X_full.T
    rss_full = np.sum((response - (H_full @ response)) ** 2)
    if X_cov.shape[1] > 0:
        X_red = np.hstack([np.ones((n, 1)), X_cov])
        H_red = X_red @ np.linalg.pinv(X_red.T @ X_red) @ X_red.T
        rss_red = np.sum((response - (H_red @ response)) ** 2)
    else:
        rss_red = np.sum((response - np.mean(response)) ** 2)
    df1 = group_dummies.shape[1]
    df2 = n - np.linalg.matrix_rank(X_full)
    ms_between = (rss_red - rss_full) / df1 if df1 > 0 else 0
    ms_within = rss_full / df2 if df2 > 0 else 0
    f_val = ms_between / ms_within if ms_within > 0 else 0
    p_val = 1.0 - f.cdf(f_val, df1, df2) if f_val > 0 else 1.0
    return f_val, p_val

def forward_feature_selection_lda(
    X: pd.DataFrame,
    y: pd.Series,
    p_thresh: float = FS_P_THRESH,
    target_num_features: int = None,
    verbose: bool = True,
    n_jobs: int = FS_N_JOBS,
    min_num_features: int = FS_MIN_NUM_FEATURES
):
    """Select features via ANCOVA-forward search."""
    selected = []
    remaining = list(X.columns)
    X_arr = X.values
    y_arr = np.array(y)
    group_dummies = pd.get_dummies(y_arr, drop_first=True).values
    feature_indices = {feat: idx for idx, feat in enumerate(X.columns)}

    def test_feature(feat):
        cov_idx = [feature_indices[f] for f in selected]
        X_cov = X_arr[:, cov_idx] if cov_idx else np.zeros((X_arr.shape[0], 0))
        response = X_arr[:, feature_indices[feat]]
        return (feat, *ancova_f_test_with_pinv(X_cov, group_dummies, response))

    stop_count = target_num_features or min_num_features
    while remaining:
        if n_jobs > 1:
            results = Parallel(n_jobs=n_jobs)(delayed(test_feature)(feat) for feat in remaining)
        else:
            results = [test_feature(feat) for feat in remaining]
        feat, f_val, p_val = min(results, key=lambda x: x[2])
        if target_num_features and len(selected) >= target_num_features:
            break
        if not target_num_features and len(selected) >= min_num_features and p_val > p_thresh:
            break
        selected.append(feat)
        remaining.remove(feat)
        if verbose:
            print(f"Selected {feat} (F={f_val:.2f}, p={p_val:.4g})")
    if verbose:
        print("Final selection:", selected)
    return selected

def run_feature_selection_methods(
    X: pd.DataFrame,
    y: pd.Series,
    methods=None,
    target_feature_counts=None,
    p_thresh: float = FS_P_THRESH,
    verbose: bool = False
):
    """Run multiple feature selection methods and return dict: method -> list of features."""
    methods = methods or FS_DEFAULT_METHODS
    if isinstance(target_feature_counts, int):
        target_feature_counts = [target_feature_counts]
    target_feature_counts = target_feature_counts or FS_TARGET_FEATURE_COUNTS
    target_feature_counts = [max(n, FS_MIN_NUM_FEATURES) for n in target_feature_counts]

    results = {}
    for m in methods:
        if m.startswith("forward"):
            if m == "forward_count":
                for n in target_feature_counts:
                    feats = forward_feature_selection_lda(X, y, target_num_features=n, verbose=verbose)
                    results[f"{m}_{n}f"] = feats
            else:
                feats = forward_feature_selection_lda(X, y, p_thresh=p_thresh, verbose=verbose)
                results[f"{m}_p{p_thresh:.3g}"] = feats

        elif m == "random_forest":
            clf = RandomForestClassifier(n_estimators=100, random_state=0)
            clf.fit(X, y)
            importances = sorted(zip(clf.feature_importances_, X.columns), reverse=True)
            for n in target_feature_counts:
                results[f"{m}_{n}f"] = [f for _, f in importances[:n]]

        elif m == "anova_kbest":
            for n in target_feature_counts:
                sel = SelectKBest(f_classif, k=n).fit(X, y)
                results[f"{m}_{n}f"] = X.columns[sel.get_support()].tolist()

        elif m == "mutual_info":
            for n in target_feature_counts:
                sel = SelectKBest(mutual_info_classif, k=n).fit(X, y)
                results[f"{m}_{n}f"] = X.columns[sel.get_support()].tolist()

        elif m == "chi2_kbest":
            for n in target_feature_counts:
                sel = SelectKBest(chi2, k=n).fit(X.abs(), y)
                results[f"{m}_{n}f"] = X.columns[sel.get_support()].tolist()

        elif m == "l1_logistic":
            clf = LogisticRegression(penalty="l1", solver="liblinear", random_state=0, max_iter=1000)
            clf.fit(X, y)
            coefs = sorted(zip(abs(clf.coef_.ravel()), X.columns), reverse=True)
            for n in target_feature_counts:
                results[f"{m}_{n}f"] = [f for _, f in coefs[:n]]

        else:
            raise ValueError(f"Unknown method '{m}'")
    return results
