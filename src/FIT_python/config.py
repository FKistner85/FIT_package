from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Data directories
DATA_DIR      = PROJECT_ROOT / "data"
RAW_DIR       = DATA_DIR / "raw"
SPLITS_DIR    = DATA_DIR / "splits"
PROCESSED_DIR = DATA_DIR / "processed"

# Results directories
RESULTS_DIR       = PROJECT_ROOT / "results"
RESULTS_DATA_DIR  = RESULTS_DIR / "data"
FIGURES_DIR       = RESULTS_DIR / "figures"

# Scripts and notebooks
SCRIPTS_DIR   = PROJECT_ROOT / "scripts"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

# Default column configurations
OTTER_META_COLS = ["id", "date", "location", "dataorigin", "substrate"]
DEFAULT_TARGETS = ["species", "individual_id", "trail", "sex"]

# Splitting parameters
GROUP_COL           = "individual_id"
STRATIFY_COL        = "sex"
GLOBAL_RANDOM_SEED  = 42
TEST_SIZE           = 0.2
NUM_FOLDS           = 5

# Misc
LOG_LEVEL = "INFO"
