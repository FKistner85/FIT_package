
"""Wrapper class for hyperparameter tuning, reading settings from config."""

from FIT_python.config import GLOBAL_RANDOM_SEED, DEBUG_MODE, # if hyperopt config exists
from FIT_python.hyperparameter_utils import optimize_hyperparameters, get_best_hyperparameters

class HyperparameterWrapper:
    def __init__(self, n_trials: int = None, seed: int = None):
        from FIT_python.config import HYPEROPTURA_TRIALS as default_trials
        self.n_trials = n_trials or default_trials
        self.seed = seed or GLOBAL_RANDOM_SEED

    def tune(
        self,
        model_name: str,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> dict:
        # Assume fold assignments and individual_ids come externally
        from FIT_python.config import # ACCORDING CONFIG VARIABLES NEEDED
        return get_best_hyperparameters(
            model_name=model_name,
            X=X,
            y=y,
            individual_ids=...,
            fold_assignments=...,
            scoring=...,
            seed=self.seed
        )
