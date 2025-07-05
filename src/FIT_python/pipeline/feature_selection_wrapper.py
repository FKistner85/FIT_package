# src/FIT_python/pipeline/feature_selection_wrapper.py

from typing import List, Optional
import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.ensemble import RandomForestClassifier

def forward_feature_selection_lda(
    X: pd.DataFrame,
    y: pd.Series,
    k: int
) -> List[str]:
    """
    ANCOVA‐basierte Forward‐Selection, wählt exakt die Top‐k Features.
    """
    selected: List[str] = []
    remaining = list(X.columns)
    X_arr = X.values
    y_arr = np.array(y)
    # One‐hot für zwei Klassen (drop_first)
    group_dummies = pd.get_dummies(y_arr, drop_first=True).values
    feature_indices = {feat: idx for idx, feat in enumerate(X.columns)}

    def score_feature(feat: str):
        i = feature_indices[feat]
        # Kovariaten = bisher ausgewählte Features
        cov_idx = [feature_indices[f] for f in selected]
        X_cov = X_arr[:, cov_idx] if cov_idx else np.zeros((X_arr.shape[0], 0))
        response = X_arr[:, i]
        # F‐Test mit Pseudoinverse
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
        # F‐Statistik
        df1 = group_dummies.shape[1]
        df2 = n - np.linalg.matrix_rank(X_full)
        msb = (rss_red - rss_full) / df1 if df1 > 0 else 0
        msw = rss_full / df2 if df2 > 0 else 0
        f_val = msb / msw if msw > 0 else 0
        # Wir verwenden den F‐Wert zum Ranking (höher = besser)
        return feat, f_val

    while len(selected) < k and remaining:
        # Score alle verbleibenden
        results = [score_feature(feat) for feat in remaining]
        # Wähle Feature mit größtem F‐Wert
        feat, _ = max(results, key=lambda x: x[1])
        selected.append(feat)
        remaining.remove(feat)

    return selected

class FeatureSelectionTransformer(TransformerMixin, BaseEstimator):
    """
    Feature Selection mit zwei Methoden:
      - 'forward': ANCOVA‐Forward, exakt k Features
      - 'random_forest': Top‐k nach RandomForest‐Importances
    """
    def __init__(
        self,
        method: str = "random_forest",
        k: int = 20,
        random_state: int = 0
    ):
        if method not in ("forward", "random_forest"):
            raise ValueError("method must be 'forward' or 'random_forest'")
        self.method = method
        self.k = k
        self.random_state = random_state
        self.selected_features_: List[str] = []

    def fit(self, X, y):
        # Stelle sicher, dass wir immer ein DataFrame haben
        if isinstance(X, pd.DataFrame):
            df = X.copy()
        else:
            df = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])

        cols = df.columns.tolist()

        if self.method == "forward":
            self.selected_features_ = forward_feature_selection_lda(df, y, k=self.k)
        else:
            arr = df.values
            rf = RandomForestClassifier(
                n_estimators=100,
                random_state=self.random_state
            ).fit(arr, y)
            importances = rf.feature_importances_
            top_idx = np.argsort(importances)[-self.k:]
            self.selected_features_ = [cols[i] for i in top_idx]

        return self

    def transform(self, X):
        # Rückgabe als NumPy-Array der ausgewählten Features
        if isinstance(X, pd.DataFrame):
            return X[self.selected_features_].values
        arr = np.asarray(X, dtype=float)
        idxs = [int(f[1:]) for f in self.selected_features_]
        return arr[:, idxs]
