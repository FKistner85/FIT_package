"""Central configuration parameters and file paths used throughout the package."""

from pathlib import Path
import os
import sys
from datetime import datetime

# ---------------------------------------------------------------------
# 1. Determine Experiment Root (always results/)
# ---------------------------------------------------------------------
_root_env = os.getenv("FIT_EXPERIMENT_ROOT")
if _root_env is not None:
    _experiment_root = Path(_root_env)
else:
    cwd = Path.cwd()
    base_root = cwd / "results"

    # Optional: wähle Experiment-Name über ENV-Variable oder Timestamp
    exp_name = os.getenv("FIT_EXPERIMENT_NAME")
    if exp_name is None:
        exp_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Jetzt klarer Pfad: results/<experiment_name>
    _experiment_root = base_root / exp_name

_experiment_root.mkdir(parents=True, exist_ok=True)

# RAW-Daten unabhängig vom Experiment
_raw_env = os.getenv("FIT_RAW_DIR")
if _raw_env is not None:
    RAW_DIR = Path(_raw_env)
else:
    RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
    if not RAW_DIR.exists():
        raise FileNotFoundError(f"RAW_DIR not found: {RAW_DIR}")

# ---------------------------------------------------------------------
# 2. Final Directories
# ---------------------------------------------------------------------
EXPERIMENT_ROOT = _experiment_root

DATA_DIR = EXPERIMENT_ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
CLEANED_DIR = DATA_DIR / "cleaned"
SPLITS_DIR = DATA_DIR / "splits"
SPLITS_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR = DATA_DIR / "processed"
PROCESSED_SPLITS_DIR = PROCESSED_DIR / "splits"
SCALED_DIR = PROCESSED_DIR / "scaled"
FEATURE_SELECTED_DIR = PROCESSED_DIR / "feature_selected"
DIM_REDUCED_DIR = PROCESSED_DIR / "dim_reduced"
NUMERIC_DIR = PROCESSED_DIR / "numeric"

FIGURES_DIR = EXPERIMENT_ROOT / "figures"
RESULTS_DATA_DIR = EXPERIMENT_ROOT / "results_data"

OTTER_LANDMARK_MAP_PATH = PROCESSED_DIR / "otter_landmark_map.json"
OTTER_POINT_MAP_PATH = PROCESSED_DIR / "otter_point_map.json"

# ---------------------------------------------------------------------
# 3. Defaults
# ---------------------------------------------------------------------
OTTER_META_COLS = ["id", "date", "location", "dataorigin", "substrate"]
DEFAULT_TARGETS = ["species", "individual_id", "trail", "sex"]

SEXMODEL_SETTINGS = {
    "search_type": "bayes_search",       # oder "random_search"
    "metric_key": "balanced_accuracy",  # oder hier sex model für pair definieren "mean_rank"
}
BAYES_REFIT = "roc_auc"

SEX_PREDICT_METRIC = "roc_auc"

GROUP_COL = "individual_id"
STRATIFY_COL = "sex"
GLOBAL_RANDOM_SEED = 12345
TEST_SIZE = 0.3
NUM_FOLDS = 5
DEBUG_MODE = False

# ---------------------------------------------------------------------
# 4. PATHS Dictionary
# ---------------------------------------------------------------------
# Only keep global locations. Species-specific directories are resolved
# dynamically via ``get_species_paths`` from ``FIT_python.utils``.
PATHS = {
    "experiment_root": EXPERIMENT_ROOT,
    "results": RESULTS_DATA_DIR,
    "figures": FIGURES_DIR,
    "raw_data": RAW_DIR,
    "pipeline_cache": RESULTS_DATA_DIR / "pipeline_cache",
}

# ---------------------------------------------------------------------
# 5. Species name mapping (Latin -> Common)
# ---------------------------------------------------------------------
SPECIES_MODEL_MAP = {
    "panthera_tigris_altaica": "amur_tiger",
    "panthera_tigris_tigris": "bengal_tiger",
    "acinonyx_jubatus": "cheetah",
    "lutra_lutra": "eurasian_otter",
    "ailuropoda_melanoleuca": "giant_panda",
    "tapirus_terrestris": "lowlandtapir",
    "puma_concolor": "mountain_lion",
    "ceratotherium_simum": "white_rhino",
}
