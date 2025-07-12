"""Preconfigured missing value imputation transformers."""

from .imputation_wrapper import ImputationWrapper

IMPUTERS = {
    "rf_default": ImputationWrapper(),
    "rf_big": ImputationWrapper(n_estimators=50, max_iter=20),
}
