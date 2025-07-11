"""Central configuration parameters and file paths used throughout the package."""

from pathlib import Path
import yaml

# Project root directory (two levels up from this file)
PROJECT_ROOT = Path(__file__).resolve().parents[2]

_CFG_PATH = Path(__file__).with_name("config.yaml")
with open(_CFG_PATH, "r", encoding="utf-8") as f:
    _CFG = yaml.safe_load(f)

def _path(name: str) -> Path:
    return PROJECT_ROOT / _CFG["paths"][name]

# Data directories
DATA_DIR = _path("data_dir")
RAW_DIR = _path("raw_dir")
CLEANED_DIR = _path("cleaned_dir")
SPLITS_DIR = _path("splits_dir")
PROCESSED_DIR = _path("processed_dir")
# Pipeline sub-directories under processed data
PROCESSED_SPLITS_DIR = _path("processed_splits_dir")
SCALED_DIR = _path("scaled_dir")
FEATURE_SELECTED_DIR = _path("feature_selected_dir")
DIM_REDUCED_DIR = _path("dim_reduced_dir")

# Numeric processed directory
NUMERIC_DIR = _path("numeric_dir")

# Processed data files
OTTER_LANDMARK_MAP_PATH = _path("otter_landmark_map_path")
OTTER_POINT_MAP_PATH = _path("otter_point_map_path")

# Results directories
RESULTS_DIR = _path("results_dir")
RESULTS_DATA_DIR = _path("results_data_dir")
FIGURES_DIR = _path("figures_dir")

# Scripts and notebooks
SCRIPTS_DIR = _path("scripts_dir")
NOTEBOOKS_DIR = _path("notebooks_dir")

# Default column configurations
OTTER_META_COLS = _CFG["data_split_and_summary"]["otter_meta_cols"]
DEFAULT_TARGETS = _CFG["data_split_and_summary"]["default_targets"]

# Splitting parameters
GROUP_COL = _CFG["data_split_and_summary"]["group_col"]
STRATIFY_COL = _CFG["data_split_and_summary"]["stratify_col"]
GLOBAL_RANDOM_SEED = _CFG["data_split_and_summary"]["global_random_seed"]
TEST_SIZE = _CFG["data_split_and_summary"]["test_size"]
NUM_FOLDS = _CFG["data_split_and_summary"]["num_folds"]

# Expose entire sub-configs for convenience
PIPELINE_SEX_CFG = _CFG.get("pipeline_sex", {})
PIPELINE_INDIVIDUAL_ID_CFG = _CFG.get("pipeline_individual_id", {})
GENERAL_PIPELINE_STEPS_CFG = _CFG.get("general_pipeline_steps", {})

#