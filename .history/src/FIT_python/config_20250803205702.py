"""Central configuration parameters and file paths used throughout the package."""

from pathlib import Path
import os
import sys

# Determine the experiment root. When the package is installed ``__file__``
# resides inside ``site-packages`` which does not contain the project data. In
# that case we fall back to the current working directory or an explicit
# environment variable ``FIT_EXPERIMENT_ROOT``. The value is only finalised once
# ``EXPERIMENT_DIR`` is known so notebooks can override it.
_BASE_ROOT = Path(__file__).resolve().parents[2]
_experiment_root = Path(os.environ.get("FIT_EXPERIMENT_ROOT", _BASE_ROOT))

# If the computed path does not contain ``data`` assume the current working
# directory is the project root. This enables running the package from a cloned
# repository without installation.
if not (_experiment_root / "data").exists():
    cwd = Path.cwd()
    if (cwd / "data").exists():
        _experiment_root = cwd

# Data and result directories are defined after EXPERIMENT_ROOT is finalised.
# They will automatically reside inside the selected experiment root so that
# each experiment keeps its own ``data`` and ``results`` folders.


# Default column configurations
OTTER_META_COLS = ["id", "date", "location", "dataorigin", "substrate"]
DEFAULT_TARGETS = ["species", "individual_id", "trail", "sex"]

# Splitting parameters
GROUP_COL = "individual_id"
STRATIFY_COL = "sex"
GLOBAL_RANDOM_SEED = 987
TEST_SIZE = 0.3
NUM_FOLDS = 5
# Global debug switch controlling fail-fast behaviour.
# True  -> raise FileNotFoundError on missing files (development)
# False -> merely log and continue (production)
DEBUG_MODE = False

from FIT_python.soft_config import SOFT_CONFIG

experiments_dir = _experiment_root / "experiments"
experiment_dir = experiments_dir / os.getenv(
    "FIT_EXPERIMENT_NAME", SOFT_CONFIG["experiment"]["name"]
)

# When executed inside a Jupyter notebook and no explicit ``FIT_EXPERIMENT_ROOT``
# is set, default to writing results into ``experiment_dir``.
if "ipykernel" in sys.modules and "FIT_EXPERIMENT_ROOT" not in os.environ:
    _experiment_root = experiment_dir

EXPERIMENT_ROOT = _experiment_root

# Paths relative to the final experiment root
NOTEBOOKS_DIR = EXPERIMENT_ROOT / "notebooks"
SCRIPTS_DIR = EXPERIMENT_ROOT / "scripts"
EXPERIMENTS_DIR = experiments_dir
EXPERIMENT_DIR = experiment_dir

DATA_DIR = EXPERIMENT_ROOT / "data"
# Raw data may live outside the experiment directory. When ``FIT_RAW_DIR`` is
# unset and ``DATA_DIR / "raw"`` does not exist, fall back to the repository
# root so notebooks executed from subfolders can still access the immutable
# dataset.
_raw_env = os.getenv("FIT_RAW_DIR")
if _raw_env is not None:
    RAW_DIR = Path(_raw_env)
else:
    _default_raw = DATA_DIR / "raw"
    if _default_raw.exists():
        RAW_DIR = _default_raw
    else:
        _root_raw = _BASE_ROOT / "data" / "raw"
        RAW_DIR = _root_raw
CLEANED_DIR = DATA_DIR / "cleaned"
SPLITS_DIR = DATA_DIR / "splits"
PROCESSED_DIR = DATA_DIR / "processed"
PROCESSED_SPLITS_DIR = PROCESSED_DIR / "splits"
SCALED_DIR = PROCESSED_DIR / "scaled"
FEATURE_SELECTED_DIR = PROCESSED_DIR / "feature_selected"
DIM_REDUCED_DIR = PROCESSED_DIR / "dim_reduced"
NUMERIC_DIR = PROCESSED_DIR / "numeric"

OTTER_LANDMARK_MAP_PATH = PROCESSED_DIR / "otter_landmark_map.json"
OTTER_POINT_MAP_PATH = PROCESSED_DIR / "otter_point_map.json"

RESULTS_DIR = EXPERIMENT_ROOT / "results"
RESULTS_DATA_DIR = RESULTS_DIR / "data"
FIGURES_DIR = RESULTS_DIR / "figures"

PATHS = {
    "pipeline_cache": RESULTS_DATA_DIR / "pipeline_cache",
    "sex_models": RESULTS_DATA_DIR / "sex_models",
    "sex_models_best": RESULTS_DATA_DIR / "sex_models_best",
    "raw_results": RESULTS_DATA_DIR / "raw_results.csv",
    "random_search": RESULTS_DATA_DIR / "random_search_standard_metrics",
    "individual_id_results": RESULTS_DATA_DIR / "individual_id_pipelines",
}

