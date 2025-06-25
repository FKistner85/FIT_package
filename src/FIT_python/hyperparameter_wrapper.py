"""Simple wrapper around Optuna-based hyperparameter optimisation utilities."""

from __future__ import annotations

import pandas as pd

from FIT_python.config import GLOBAL_RANDOM_SEED, HYPER_N_TRIALS
from FIT_python.hyperparameter_utils import get_best_hyperparameters


class HyperparameterWrapper:
    """Expose a stable interface used by other wrappers."""

    def __init__(self, n_trials: int | None = None, seed: int | None = None) -> None:
        self.n_trials = n_trials or HYPER_N_TRIALS
        self.seed = seed or GLOBAL_RANDOM_SEED

    def tune(
        self,
        model_name: str,
        X: pd.DataFrame,
        y: pd.Series,
        individual_ids: pd.Series,
        fold_assignments: pd.DataFrame,
        scoring: str,
    ) -> dict:
        """Return the best hyperparameters for the given data."""
        return get_best_hyperparameters(
            model_name=model_name,
            X=X,
            y=y,
            individual_ids=individual_ids,
            fold_assignments=fold_assignments,
            scoring=scoring,
            seed=self.seed,
        )
