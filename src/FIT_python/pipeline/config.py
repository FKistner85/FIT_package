"""Configuration values for the ML pipeline."""

from FIT_python.config import (
    SPLITS_DIR,
    RESULTS_DATA_DIR,
    FIGURES_DIR,
    GLOBAL_RANDOM_SEED,
    NUM_FOLDS,
    GROUP_COL,
)

from .models import MODELS

# Available options
FS_METHODS = ["forward", "random_forest", "variance", "univariate", "lasso"]
IMPUTE_OPTIONS = ["miss_forest"]
OUTLIER_OPTIONS = ["clip", "zscore"]
SCALER_OPTIONS = ["standard", "robust"]
DIMRED_OPTIONS = ["pca"]
MODEL_KEYS = list(MODELS.keys())
