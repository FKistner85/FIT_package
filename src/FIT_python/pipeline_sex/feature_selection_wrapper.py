import numpy as np
import pandas as pd
from typing import List, Tuple, Union
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif, VarianceThreshold
from sklearn.linear_model import LassoCV


def _forward_ranking(
    X_arr: np.ndarray, y_arr: np.ndarray, feature_names: List[str], k_max: int
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
    def __init__(
        self,
        method: str = None,  # 'forward', 'random_forest', 'variance', 'univariate', 'lasso', or None (use all)
        k: int = None,
        random_state: int = 0,
    ):
        allowed_methods = [
            None,
            "forward",
            "random_forest",
            "variance",
            "univariate",
            "lasso",
        ]
        if method not in allowed_methods:
            raise ValueError(f"method must be one of {allowed_methods}")
        self.method = method
        self.k = k
        self.random_state = random_state
        self.feature_ranking_: List[Tuple[str, float]] = []
        self.selected_features_: List[str] = []

    def fit(self, X: Union[pd.DataFrame, np.ndarray], y):
        if isinstance(X, pd.DataFrame):
            df = X.copy()
            feat_names = df.columns.tolist()
            arr = df.values
        else:
            arr = np.asarray(X, float)
            feat_names = [f"f{i}" for i in range(arr.shape[1])]

        k_max = self.k or arr.shape[1]

        if self.method is None:
            self.selected_features_ = feat_names  # ALLE Features verwenden
            self.feature_ranking_ = [(f, 1.0) for f in feat_names]

        elif self.method == "forward":
            self.feature_ranking_ = _forward_ranking(
                arr, np.array(y), feat_names, k_max
            )

        elif self.method == "random_forest":
            rf = RandomForestClassifier(
                n_estimators=100, random_state=self.random_state
            ).fit(arr, y)
            imp = rf.feature_importances_
            ranked = sorted(zip(feat_names, imp), key=lambda x: x[1], reverse=True)
            self.feature_ranking_ = ranked[:k_max]

        elif self.method == "variance":
            vt = VarianceThreshold()
            vt.fit(arr)
            variances = vt.variances_
            ranked = sorted(
                zip(feat_names, variances), key=lambda x: x[1], reverse=True
            )
            self.feature_ranking_ = ranked[:k_max]

        elif self.method == "univariate":
            skb = SelectKBest(score_func=f_classif, k="all").fit(arr, y)
            scores = skb.scores_
            ranked = sorted(
                zip(feat_names, scores),
                key=lambda x: x[1] if x[1] is not None else 0,
                reverse=True,
            )
            self.feature_ranking_ = ranked[:k_max]

        elif self.method == "lasso":
            lasso = LassoCV(cv=5, random_state=self.random_state).fit(arr, y)
            imp = np.abs(lasso.coef_)
            ranked = sorted(zip(feat_names, imp), key=lambda x: x[1], reverse=True)
            self.feature_ranking_ = ranked[:k_max]

        k_use = self.k or len(self.feature_ranking_)
        self.selected_features_ = [feat for feat, _ in self.feature_ranking_[:k_use]]
        return self

    def transform(self, X: Union[pd.DataFrame, np.ndarray]):
        if isinstance(X, pd.DataFrame):
            return X[self.selected_features_].values
        else:
            arr = np.asarray(X, float)
            all_feat_names = [f"f{i}" for i in range(arr.shape[1])]
            selected_indices = [
                all_feat_names.index(f) for f in self.selected_features_
            ]
            return arr[:, selected_indices]


# Shortcuts combining method and ``k`` to limit hyperparameter search
FEATURE_SELECTION_PRESETS = {
    "forward_10": FeatureSelectionTransformer(method="forward", k=10),
    "random_forest_20": FeatureSelectionTransformer(method="random_forest", k=20),
    "variance": FeatureSelectionTransformer(method="variance"),
    "lasso_10": FeatureSelectionTransformer(method="lasso", k=10),
}
