"""Central configuration parameters and file paths used throughout the package."""

from pathlib import Path
import os

# Determine the experiment root. When the package is installed, ``__file__``
# points inside ``site-packages`` which does not contain the project data.  In
# that case we fall back to the current working directory or an explicit
# environment variable ``FIT_EXPERIMENT_ROOT``.
EXPERIMENT_ROOT = Path(
    os.environ.get("FIT_EXPERIMENT_ROOT", Path(__file__).resolve().parents[2])
)

# If the computed path does not contain the ``data`` directory, assume the
# current working directory is the experiment root.  This enables running the
# package from a cloned repository without installation.
if not (EXPERIMENT_ROOT / "data").exists():
    cwd = Path.cwd()
    if (cwd / "data").exists():
        EXPERIMENT_ROOT = cwd

# Data directories
DATA_DIR = EXPERIMENT_ROOT / "data"
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

# Unified file paths for generated artefacts
PATHS = {
    "pipeline_cache": RESULTS_DATA_DIR / "pipeline_cache",
    "sex_models": RESULTS_DATA_DIR / "sex_models",
    "sex_models_best": RESULTS_DATA_DIR / "sex_models_best",
    "raw_results": RESULTS_DATA_DIR / "raw_results.csv",
    "random_search": RESULTS_DATA_DIR / "random_search_standard_metrics",
    "individual_id_results": RESULTS_DATA_DIR / "individual_id_pipelines",
}

# Results directories
RESULTS_DIR = EXPERIMENT_ROOT / "results"
RESULTS_DATA_DIR = RESULTS_DIR / "data"
FIGURES_DIR = RESULTS_DIR / "figures"

# Scripts and notebooks
SCRIPTS_DIR = EXPERIMENT_ROOT / "scripts"
NOTEBOOKS_DIR = EXPERIMENT_ROOT / "notebooks"

# Default column configurations
OTTER_META_COLS = ["id", "date", "location", "dataorigin", "substrate"]
DEFAULT_TARGETS = ["species", "individual_id", "trail", "sex"]

# Splitting parameters
GROUP_COL = "individual_id"
STRATIFY_COL = "sex"
GLOBAL_RANDOM_SEED = 99
TEST_SIZE = 0.2
NUM_FOLDS = 3
# Global debug switch controlling fail-fast behaviour.
# True  -> raise FileNotFoundError on missing files (development)
# False -> merely log and continue (production)
DEBUG_MODE = False
