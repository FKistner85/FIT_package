from pathlib import Path

# Project root directory (two levels up from this file)
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
CLEANED_DIR = DATA_DIR / "cleaned"
SPLITS_DIR = DATA_DIR / "splits"
PROCESSED_DIR = DATA_DIR / "processed"
# Pipeline sub-directories under processed data
PROCESSED_SPLITS_DIR = PROCESSED_DIR / "splits"
SCALED_DIR = PROCESSED_DIR / "scaled"
FEATURE_SELECTED_DIR = PROCESSED_DIR / "feature_selected"
DIM_REDUCED_DIR = PROCESSED_DIR / "dim_reduced"

# Numeric processed directory
NUMERIC_DIR = PROCESSED_DIR / "numeric"

# Processed data files
OTTER_LANDMARK_MAP_PATH = PROCESSED_DIR / "otter_landmark_map.json"
OTTER_POINT_MAP_PATH = PROCESSED_DIR / "otter_point_map.json"

# Results directories
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DATA_DIR = RESULTS_DIR / "data"
FIGURES_DIR = RESULTS_DIR / "figures"

# Scripts and notebooks
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

# Default column configurations
OTTER_META_COLS = ["id", "date", "location", "dataorigin", "substrate"]
DEFAULT_TARGETS = ["species", "individual_id", "trail", "sex"]

# Splitting parameters
GROUP_COL = "individual_id"
STRATIFY_COL = "sex"
GLOBAL_RANDOM_SEED = 42
TEST_SIZE = 0.2
NUM_FOLDS = 3

# Scaling configuration
DEFAULT_SCALER = "standard"
SCALER_PARAMS = {"standard": {}, "robust": {}}

# Misc
LOG_LEVEL = "INFO"

# Global debug switch controlling fail-fast behaviour
# True  -> raise FileNotFoundError on missing files (development)
# False -> merely log and continue (production)
DEBUG_MODE = True

# -------------------------------------------------------------------
# Dataset name normalization


def normalize_dataset_name(name: str) -> str:
    """Return a normalized dataset name.

    Spaces are replaced with underscores and the result is lowercased so that
    all pipeline steps use consistent folder names.
    """
    return name.strip().replace(" ", "_").lower()


# -------------------------------------------------------------------
# Feature Selection parameters

# Welche Methoden sollen standardmäßig ausgeführt werden
FS_DEFAULT_METHODS = [
    "forward_count",  # Forward‐Selektion mit fester Anzahl Features
    "forward_p",  # Forward‐Selektion basierend auf p‐Wert
    "random_forest",  # Wichtigkeit aus RandomForest
    "anova_kbest",  # SelectKBest mit ANOVA F-Test
    "mutual_info",  # SelectKBest mit Mutual Information
    "chi2_kbest",  # SelectKBest mit Chi-Quadrat
    "l1_logistic",  # L1-Regularisierung (Logistic Regression)
]

# Standard‐Anzahl von Features, die pro Methode ausgewählt werden
FS_TARGET_FEATURE_COUNTS = [10, 20, 30]

# Schwellen für ANCOVA-Forward-Selection
FS_P_THRESH = 0.05  # Stoppe, wenn p > 0.05 nach mindestens min_num_features
FS_MIN_NUM_FEATURES = 3  # Minimal immer so viele Features einfügen
FS_N_JOBS = 1  # Anzahl paralleler Prozesse für die Selektion
# Validation parameters
# "external_folds": use precomputed grouped/stratified folds
# "internal_cv":    use sklearn GridSearchCV with cv=3
VALIDATION_MODE = "external_folds"
NUM_KFOLDS = 3

# -------------------------------------------------------------------
# Simplified pipeline parameters

# Number of top-variance features to retain in the simple feature
# selection script.
N_FEATURES = 2

# Number of principal components produced by the basic PCA reduction
# step.
N_COMPONENTS = 2

# -------------------------------------------------------------------
# Advanced pipeline configuration

# Dimensionality reduction methods to evaluate. If empty, all available
# methods from ``dim_reduction_utils`` are used.
DIM_REDUCTION_METHODS = ["pca"]

# Models to compare in the sex classification pipeline. Only very small
# defaults are provided here for demonstration purposes.
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

MODELS = {
    "logreg": LogisticRegression(max_iter=1000, solver="liblinear"),
    "svc": SVC(),
}

# Hyper-parameter grids per model and target.
PIPELINE_PARAM_GRIDS = {
    "sex": {
        "logreg": {"clf__C": [0.1, 1, 10]},
        "svc": {"clf__C": [0.1, 1, 10], "clf__kernel": ["linear", "rbf"]},
    }
}

# Metric used for optimisation during grid-search.
METRICS = {"sex": "accuracy"}

# Targets supported by the pipeline scripts.
PIPELINE_TARGETS = ["sex"]
