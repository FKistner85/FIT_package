"""Central configuration parameters and file paths used throughout the package."""

from pathlib import Path
import os
import sys

# ---------------------------------------------------------------------
# 1. Determine Experiment Root (one level below the notebook location)
# ---------------------------------------------------------------------

# Resolve current working dir (Jupyter or script)
cwd = Path.cwd()

# One level deeper than the notebook directory
_experiment_root = cwd / "results"

# Allow explicit override via environment variable
_exp_env = os.getenv("FIT_EXPERIMENT_ROOT")
if _exp_env is not None:
    _experiment_root = Path(_exp_env)

# ---------------------------------------------------------------------
# 2. RAW Data directory (independent of experiment root)
# ---------------------------------------------------------------------
_raw_env = os.getenv("FIT_RAW_DIR")
if _raw_env is not None:
    RAW_DIR = Path(_raw_env)
else:
    RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
    if not RAW_DIR.exists():
        raise FileNotFoundError(f"RAW_DIR not found: {RAW_DIR}")

# ---------------------------------------------------------------------
# 3. Final Directories
# ---------------------------------------------------------------------
EXPERIMENT_ROOT = _experiment_root

RESULTS_DIR = EXPERIMENT_ROOT
DATA_DIR = RESULTS_DIR / "data"
CLEANED_DIR = DATA_DIR / "cleaned"
SPLITS_DIR = DATA_DIR / "splits"
PROCESSED_DIR = DATA_DIR / "processed"
PROCESSED_SPLITS_DIR = PROCESSED_DIR / "splits"
SCALED_DIR = PROCESSED_DIR / "scaled"
FEATURE_SELECTED_DIR = PROCESSED_DIR / "feature_selected"
DIM_REDUCED_DIR = PROCESSED_DIR / "dim_reduced"
NUMERIC_DIR = PROCESSED_DIR / "numeric"

FIGURES_DIR = RESULTS_DIR / "figures"
RESULTS_DATA_DIR = RESULTS_DIR / "results_data"

OTTER_LANDMARK_MAP_PATH = PROCESSED_DIR / "otter_landmark_map.json"
OTTER_POINT_MAP_PATH = PROCESSED_DIR / "otter_point_map.json"

# ---------------------------------------------------------------------
# 4. Default Configs
# ---------------------------------------------------------------------
OTTER_META_COLS = ["id", "date", "location", "dataorigin", "substrate"]
DEFAULT_TARGETS = ["species", "individual_id", "trail", "sex"]

GROUP_COL = "individual_id"
STRATIFY_COL = "sex"
GLOBAL_RANDOM_SEED = 987
TEST_SIZE = 0.3
NUM_FOLDS = 5
DEBUG_MODE = False

# ---------------------------------------------------------------------
# 5. Paths Dictionary
# ---------------------------------------------------------------------
PATHS = {
    "pipeline_cache": RESULTS_DATA_DIR / "pipeline_cache",
    "sex_models": RESULTS_DATA_DIR / "sex_models",
    "sex_models_best": RESULTS_DATA_DIR / "sex_models_best",
    "raw_results": RESULTS_DATA_DIR / "raw_results.csv",
    "random_search": RESULTS_DATA_DIR / "random_search_standard_metrics",
    "individual_id_results": RESULTS_DATA_DIR / "individual_id_pipelines",
}
