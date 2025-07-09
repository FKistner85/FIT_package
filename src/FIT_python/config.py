"""Central configuration parameters and file paths used throughout the package."""

from pathlib import Path

# Project root directory (two levels up from this file)
PROJECT_ROOT = Path(__file__).parents[2]

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
GLOBAL_RANDOM_SEED = 123
TEST_SIZE = 0.2
NUM_FOLDS = 3
# Global debug switch controlling fail-fast behaviour.
# True  -> raise FileNotFoundError on missing files (development)
# False -> merely log and continue (production)
DEBUG_MODE = False
