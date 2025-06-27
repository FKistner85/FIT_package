import numpy as np
import pandas as pd
from typing import List, Tuple, Union
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.ensemble import RandomForestClassifier


def _forward_ranking(
    X_arr: np.ndarray,
    y_arr: np.ndarray,
    feature_names: List[str],
    k_max: int
) -> List[Tuple[str, float]]:
    """
    ANCOVA-forward search bis k_max Features:
    liefert eine Liste (feat, F-Wert) in Selektionsreihenfolge.
    """
    selected: List[str] = []
    ranking: List[Tuple[str, float]] = []
    remaining = list(feature_names)
    n = len(y_arr)

    # Dummy-Codierung für Gruppen (2 Klassen, drop_first)
    group_dummies = pd.get_dummies(y_arr, drop_first=True).values
    feat_idx = {feat: i for i, feat in enumerate(feature_names)}

    def score(feat: str) -> float:
        idx = feat_idx[feat]
        # Kovariatenmatrix
        if selected:
            cov_idx = [feat_idx[f] for f in selected]
            X_cov = X_arr[:, cov_idx]
        else:
            X_cov = np.zeros((n, 0))

        resp = X_arr[:, idx]
        # Volles Modell
        X_full = np.hstack([np.ones((n, 1)), X_cov, group_dummies])
        H_full = X_full @ np.linalg.pinv(X_full.T @ X_full) @ X_full.T
        rss_full = ((resp - H_full @ resp) ** 2).sum()

        # Reduziertes Modell
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

    # iterativ Features auswählen bis k_max
    while len(ranking) < k_max and remaining:
        scores = [(feat, score(feat)) for feat in remaining]
        best_feat, best_score = max(scores, key=lambda x: x[1])
        ranking.append((best_feat, best_score))
        remaining.remove(best_feat)
        selected.append(best_feat)

    return ranking


class FeatureSelectionTransformer(TransformerMixin, BaseEstimator):
    """
    Feature Selection mit zwei Methoden:
     - 'forward': ANCOVA-Forward bis k_max
     - 'random_forest': Top-k_max via RandomForest-Importances

    Anschließend wählbar für beliebiges k ≤ k_max.
    """
    def __init__(
        self,
        method: str = "random_forest",
        k: int = None,
        random_state: int = 0
    ):
        if method not in ("forward", "random_forest"):
            raise ValueError("method must be 'forward' or 'random_forest'")
        self.method = method
        self.k = k  # für finale Auswahl
        self.random_state = random_state

        # wird in fit() gefüllt:
        self.feature_ranking_: List[Tuple[str, float]] = []
        self.selected_features_: List[str] = []

    def fit(self, X: Union[pd.DataFrame, np.ndarray], y):
        # vereinheitliche Array und Feature-Namen
        if isinstance(X, pd.DataFrame):
            df = X.copy()
            feat_names = df.columns.tolist()
            arr = df.values
        else:
            arr = np.asarray(X, float)
            feat_names = [f"f{i}" for i in range(arr.shape[1])]

        # k_max festlegen (None → alle Features)
        k_max = self.k or arr.shape[1]

        if self.method == "forward":
            # ANCOVA-Forward Ranking
            self.feature_ranking_ = _forward_ranking(
                arr, np.array(y), feat_names, k_max
            )
        else:
            # RandomForest-Importances Ranking
            rf = RandomForestClassifier(
                n_estimators=100, random_state=self.random_state
            ).fit(arr, y)
            imp = rf.feature_importances_
            ranked = sorted(
                zip(feat_names, imp),
                key=lambda x: x[1],
                reverse=True
            )
            self.feature_ranking_ = ranked[:k_max]

        # finale Auswahl für aktuelles k
        k_use = self.k or len(self.feature_ranking_)
        self.selected_features_ = [
            feat for feat, _ in self.feature_ranking_[:k_use]
        ]
        return self

    def transform(self, X: Union[pd.DataFrame, np.ndarray]):
        # nur ausgewählte Merkmale zurückgeben
        if isinstance(X, pd.DataFrame):
            return X[self.selected_features_].values
        arr = np.asarray(X, float)
        # wenn Namen f{i}, extrahiere Index
        idxs = [
            int(f[1:]) if f.startswith("f") else i
            for i, f in enumerate(self.selected_features_)
        ]
        return arr[:, idxs]
