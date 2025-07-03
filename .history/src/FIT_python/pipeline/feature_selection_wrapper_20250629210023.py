import numpy as np
import pandas as pd
from typing import List, Tuple, Union
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif, VarianceThreshold
from sklearn.linear_model import LassoCV


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
