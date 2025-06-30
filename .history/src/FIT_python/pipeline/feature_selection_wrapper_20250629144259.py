import numpy as np
import pandas as pd
from typing import List, Tuple, Union, Dict
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.ensemble import RandomForestClassifier


def _forward_ranking(
    X_arr: np.ndarray,
    y_arr: np.ndarray,
    feature_names: List[str],
    k_max: int
) -> List[Tuple[str, float]]:
    selected: List[str] = []
    ranking: List[Tuple[str, float]] = []
    remaining = list(feature_names)
    n = len(y_arr)

    group_dummies = pd.get_dummies(y_arr, drop_first=True).values
    feat_idx = {feat: i for i, feat in enumerate(feature_names)}

    def score(feat: str) -> float:
        idx = feat_idx[feat]
        if selected:
            cov_idx = [feat_idx[f] for f in selected]
            X_cov = X_arr[:, cov_idx]
        else:
            X_cov = np.zeros((n, 0))

        resp = X_arr[:, idx]
        X_full = np.hstack([np.ones((n, 1)), X_cov, group_dummies])
        H_full = X_full @ np.linalg.pinv(X_full.T @ X_full) @ X_full.T
        rss_full = ((resp - H_full @ resp) ** 2).sum()

        if X_cov.shape[1] > 0:
            X_red = np.hstack([np.ones((n, 1)), X_cov])
            H_red = X_red @ np.linalg.pinv(X_red.T @ X_red) @ X_red.T
            rss_red = ((resp - H_red @ resp) ** 2).sum()
        else:
            rss_red = ((resp - resp.mean()) ** 2).sum()

        df1 = group_dummies.shape[1]
        df2 = n - np.linalg.matrix_rank(X_full)
        msb = (rss_red - rss_full) / (df1 or 1)
        msw = rss_full / (df2 or 1)
        return msb / msw if msw > 0 else 0.0

    while len(ranking) < k_max and remaining:
        scores = [(feat, score(feat)) for feat in remaining]
        best_feat, best_score = max(scores, key=lambda x: x[1])
        ranking.append((best_feat, best_score))
        remaining.remove(best_feat)
        selected.append(best_feat)

    return ranking


class FeatureSelectionTransformer(TransformerMixin, BaseEstimator):
    """
    Feature Selection Transformer supporting:
    - 'forward': ANCOVA-based forward selection
    - 'random_forest': Top-k by feature importance
    - 'all': Computes rankings for all methods
    """
    def __init__(
        self,
        method: str = "random_forest",  # "forward", "random_forest", or "all"
        k: int = None,                  # Number of features (None = all)
        random_state: int = 0
    ):
        if method not in ("forward", "random_forest", "all"):
            raise ValueError("method must be 'forward', 'random_forest' or 'all'")
        self.method = method
        self.k = k
        self.random_state = random_state

        # Results
        self.feature_ranking_: Dict[str, List[Tuple[str, float]]] = {}
        self.selected_features_: Dict[str, List[str]] = {}
        self.active_method: str = None  # Will be used in transform()

    def fit(self, X: Union[pd.DataFrame, np.ndarray], y):
        if isinstance(X, pd.DataFrame):
            arr = X.values
            feat_names = X.columns.tolist()
        else:
            arr = np.asarray(X, float)
            feat_names = [f"f{i}" for i in range(arr.shape[1])]

        k_max = self.k or arr.shape[1]
        methods = ["forward", "random_forest"] if self.method == "all" else [self.method]

        for method in methods:
            if method == "forward":
                rank = _forward_ranking(arr, np.array(y), feat_names, k_max)
            else:
                rf = RandomForestClassifier(n_estimators=100, random_state=self.random_state).fit(arr, y)
                imp = rf.feature_importances_
                rank = sorted(zip(feat_names, imp), key=lambda x: x[1], reverse=True)[:k_max]

            self.feature_ranking_[method] = rank
            self.selected_features_[method] = [feat for feat, _ in rank[:self.k or len(rank)]]

        # Set active method to the first one if multiple
        self.active_method = methods[0]
        return self

    def transform(self, X: Union[pd.DataFrame, np.ndarray]):
        if self.active_method is None:
            raise ValueError("No feature selection method was fitted.")

        selected_feats = self.selected_features_[self.active_method]

        if isinstance(X, pd.DataFrame):
            return X[selected_feats].values
        else:
            arr = np.asarray(X, float)
            feat_idx = [int(f[1:]) if f.startswith("f") else i for i, f in enumerate(selected_feats)]
            return arr[:, feat_idx]

    def set_active_method(self, method: str):
        if method not in self.selected_features_:
            raise ValueError(f"Method '{method}' not found in fitted selector.")
        self.active_method = method
