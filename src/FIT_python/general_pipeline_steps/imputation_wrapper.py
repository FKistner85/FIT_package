# src/FIT_python/pipeline/imputation_wrapper.py

import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor
from FIT_python.soft_config import SOFT_CONFIG
from FIT_python.utils import debug_report


class ImputationWrapper(TransformerMixin, BaseEstimator):
    """Impute missing numeric values using ``IterativeImputer``.

    The wrapper configures :class:`~sklearn.impute.IterativeImputer` with a
    :class:`~sklearn.ensemble.RandomForestRegressor` estimator. Default values
    for ``n_estimators``, ``max_iter`` and ``random_state`` are taken from the
    :data:`SOFT_CONFIG` dictionary.
    """

    def __init__(
        self,
        n_estimators: int = SOFT_CONFIG["general_pipeline_steps"][
            "imputation_defaults"
        ]["n_estimators"],
        max_iter: int = SOFT_CONFIG["general_pipeline_steps"]["imputation_defaults"][
            "max_iter"
        ],
        random_state: int = SOFT_CONFIG["general_pipeline_steps"][
            "imputation_defaults"
        ]["random_state"],
    ):
        """Initialise the wrapper with RandomForest-based imputation.

        Parameters
        ----------
        n_estimators:
            Number of trees for the underlying ``RandomForestRegressor``.
        max_iter:
            Maximum number of imputation rounds performed by
            ``IterativeImputer``.
        random_state:
            Seed used for both the regressor and the imputer.
        """
        # 1) Signatur-Parameter als Attribute setzen
        self.n_estimators = n_estimators
        self.max_iter = max_iter
        self.random_state = random_state

        # 2) Den eigentlichen Imputer bauen
        self.imputer = IterativeImputer(
            estimator=RandomForestRegressor(
                n_estimators=n_estimators, random_state=random_state
            ),
            max_iter=max_iter,
            initial_strategy="median",
            random_state=random_state,
        )

    def fit(self, X, y=None):
        """Fit the underlying imputer on the numeric columns of ``X``.

        Parameters
        ----------
        X:
            Input data as ``pandas.DataFrame`` or ``numpy.ndarray``.
        y:
            Ignored, present for compatibility with sklearn's API.
        """
        # Unterscheide DataFrame vs. np.ndarray
        if isinstance(X, pd.DataFrame):
            X_num = X.select_dtypes(include=[np.number])
        else:
            X_num = np.asarray(X, dtype=float)
        self.imputer.fit(X_num)
        return self

    def transform(self, X):
        """Return ``X`` with missing numeric values imputed."""
        if isinstance(X, pd.DataFrame):
            X_out = X.copy()
            num_cols = X_out.select_dtypes(include=[np.number]).columns
            X_out[num_cols] = self.imputer.transform(X_out[num_cols])
            debug_report(X_out, "impute")
            return X_out
        else:
            arr = np.asarray(X, dtype=float)
            X_out = self.imputer.transform(arr)
            debug_report(X_out, "impute")
            return X_out
