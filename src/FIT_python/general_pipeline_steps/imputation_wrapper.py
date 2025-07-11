# src/FIT_python/pipeline/imputation_wrapper.py

import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor

class ImputationWrapper(TransformerMixin, BaseEstimator):
    def __init__(self, n_estimators=10, max_iter=10, random_state=42):
        # 1) Signatur-Parameter als Attribute setzen
        self.n_estimators = n_estimators
        self.max_iter     = max_iter
        self.random_state = random_state

        # 2) Den eigentlichen Imputer bauen
        self.imputer = IterativeImputer(
            estimator=RandomForestRegressor(
                n_estimators=n_estimators,
                random_state=random_state
            ),
            max_iter=max_iter,
            initial_strategy='median',
            random_state=random_state
        )
    def fit(self, X, y=None):
        # Unterscheide DataFrame vs. np.ndarray
        if isinstance(X, pd.DataFrame):
            X_num = X.select_dtypes(include=[np.number])
        else:
            X_num = np.asarray(X, dtype=float)
        self.imputer.fit(X_num)
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            X_out = X.copy()
            num_cols = X_out.select_dtypes(include=[np.number]).columns
            X_out[num_cols] = self.imputer.transform(X_out[num_cols])
            return X_out
        else:
            arr = np.asarray(X, dtype=float)
            return self.imputer.transform(arr)
