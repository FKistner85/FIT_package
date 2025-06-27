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

# Feature selection defaults (for legacy wrappers)
FS_P_THRESH = 0.05
FS_MIN_NUM_FEATURES = 3
FS_N_JOBS = 1
FS_DEFAULT_METHODS = [
    "forward_count",
    "forward_p",
    "random_forest",
    "anova_kbest",
    "mutual_info",
    "chi2_kbest",
    "l1_logistic",
]
FS_TARGET_FEATURE_COUNTS = [10, 20, 30]

# Scaling defaults
DEFAULT_SCALER = "standard"
SCALER_PARAMS = {"standard": {}, "robust": {}}

# Misc settings
DEBUG_MODE = True


def normalize_dataset_name(name: str) -> str:
    """Normalize dataset folder names consistently."""
    return name.strip().replace(" ", "_").lower()

#