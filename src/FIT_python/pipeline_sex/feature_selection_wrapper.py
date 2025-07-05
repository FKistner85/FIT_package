import numpy as np
import pandas as pd
from typing import List, Tuple, Union
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif, VarianceThreshold
from sklearn.linear_model import LassoCV


def _forward_ranking(
    X_arr: np.ndarray,
    y_arr: np.ndarray,
    feature_names: List[str],
    k_max: int,
) -> List[Tuple[str, float]]:
    """Greedy forward selection using vectorised F-statistics."""

    n = len(y_arr)
    group_dummies = pd.get_dummies(y_arr, drop_first=True).values.astype(float)

    feat_idx = {feat: i for i, feat in enumerate(feature_names)}
    remaining = list(feature_names)
    selected: list[str] = []
    ranking: list[Tuple[str, float]] = []

    while len(ranking) < k_max and remaining:
        sel_idx = [feat_idx[f] for f in selected]
        X_sel = X_arr[:, sel_idx] if sel_idx else np.empty((n, 0))
        Z_red = np.hstack([np.ones((n, 1)), X_sel])
        Z_full = np.hstack([Z_red, group_dummies])

        pinv_red = np.linalg.pinv(Z_red)
        pinv_full = np.linalg.pinv(Z_full)

        cand_idx = [feat_idx[f] for f in remaining]
        Y = X_arr[:, cand_idx]

        res_full = Y - Z_full @ (pinv_full @ Y)
        rss_full = np.sum(res_full * res_full, axis=0)

        res_red = Y - Z_red @ (pinv_red @ Y)
        rss_red = np.sum(res_red * res_red, axis=0)

        df1 = group_dummies.shape[1]
        df2 = n - np.linalg.matrix_rank(Z_full)

        msb = (rss_red - rss_full) / (df1 or 1)
        msw = rss_full / (df2 or 1)
        f_scores = np.divide(msb, msw, out=np.zeros_like(msb), where=msw > 0)

        best_pos = int(np.argmax(f_scores))
        best_feat = remaining.pop(best_pos)
        best_score = float(f_scores[best_pos])

        ranking.append((best_feat, best_score))
        selected.append(best_feat)

    return ranking


class FeatureSelectionTransformer(TransformerMixin, BaseEstimator):
    def __init__(
        self,
        method: str = None,  # 'forward', 'random_forest', 'variance', 'univariate', 'lasso', or None (use all)
        k: int = None,
        random_state: int = 0
    ):
        allowed_methods = [None, 'forward', 'random_forest', 'variance', 'univariate', 'lasso']
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
            self.feature_ranking_ = _forward_ranking(arr, np.array(y), feat_names, k_max)

        elif self.method == "random_forest":
            rf = RandomForestClassifier(n_estimators=100, random_state=self.random_state).fit(arr, y)
            imp = rf.feature_importances_
            ranked = sorted(zip(feat_names, imp), key=lambda x: x[1], reverse=True)
            self.feature_ranking_ = ranked[:k_max]

        elif self.method == "variance":
            vt = VarianceThreshold()
            vt.fit(arr)
            variances = vt.variances_
            ranked = sorted(zip(feat_names, variances), key=lambda x: x[1], reverse=True)
            self.feature_ranking_ = ranked[:k_max]

        elif self.method == "univariate":
            skb = SelectKBest(score_func=f_classif, k='all').fit(arr, y)
            scores = skb.scores_
            ranked = sorted(zip(feat_names, scores), key=lambda x: x[1] if x[1] is not None else 0, reverse=True)
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
            selected_indices = [all_feat_names.index(f) for f in self.selected_features_]
            return arr[:, selected_indices]
