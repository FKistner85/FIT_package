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
BAYES_REFIT = "balanced_accuracy"


# Default metric used when selecting the best sex-classification model.
# Must match one of the keys in ``SCORING`` defined in

# :mod:`FIT_python.pipeline_sex.sex_config`.
SEX_PREDICT_METRIC = "balanced_accuracy"



# ---------------------------------------------------------------------
# 3b. Visualisation defaults
# ---------------------------------------------------------------------


def _lighten(color: str, amount: float) -> str:
    """Return a lighter shade of ``color``.

    ``amount`` specifies the blend ratio with white where ``0`` returns the
    original colour and ``1`` returns white.
    """

    color = color.lstrip("#")
    r = int(color[0:2], 16) / 255
    g = int(color[2:4], 16) / 255
    b = int(color[4:6], 16) / 255
    r = round((r + (1 - r) * amount) * 255)
    g = round((g + (1 - g) * amount) * 255)
    b = round((b + (1 - b) * amount) * 255)
    return f"#{r:02x}{g:02x}{b:02x}"


SEX_VALUE_MAP = {
    "f": "Female",
    "F": "Female",
    0: "Female",
    "m": "Male",
    "M": "Male",
    1: "Male",
}

SEX_COLORS = {
    "Female": "#800000",  # dark red
    "Male": "#000080",  # navy
    "Unknown": "#FFA500",  # orange
}

TRAIN_COLORS = SEX_COLORS
TEST_COLORS = {k: _lighten(v, 0.5) for k, v in SEX_COLORS.items()}


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
# ---------------------------------------------------------------------
# 6. Consolidated configuration
# ---------------------------------------------------------------------

# Global configuration previously spread across multiple modules.
# Access this dictionary to configure pipelines and helpers.
CONFIG = {
    "data_split_and_summary": {
        "sex_categories": ["Female", "Male"],
        "split_labels": ["Train", "Test"],
        "species_remap": {
            "a_j_soemmeringii": "acinonyx_jubatus_soemmeringii",
            "a_j_jubatus": "acinonyx_jubatus_jubatus",
        },
    },
    "general_pipeline_steps": {
        "outlier_defaults": {
            "method": "clip",
            "lower_quantile": 0.01,
            "upper_quantile": 0.99,
            "z_thresh": 3.0,
        },
        "scaler_default": {"method": "standard"},
        "imputation_defaults": {
            "n_estimators": 10,
            "max_iter": 10,
            "random_state": GLOBAL_RANDOM_SEED,
        },
        "dim_reducer_defaults": {
            "method": "pca",
            "n_components": 2,
            "n_neighbors": 15,
            "min_dist": 0.1,
            "whiten": False,
        },
        # Columns that should be preserved when wrappers operate on DataFrames
        # regardless of their data type. Pipelines may overwrite this entry at
        # runtime to customise behaviour.
        "metadata_cols": ["individual_id", "Trail", "sex", "id", "Fold"],
    },
    "pipeline_sex": {
        "model_keys": [
            "logreg_l2",
            "logreg_l1",
            "rf_small",
            "rf_med",
            "rf_large",
            "knn_3",
            "knn_5",
            "knn_7",
            "svm_linear",
            "svm_rbf",
            "xgb_std",
            "xgb_hist",
            "lgbm_std",
            "lgbm_md10",
            "lda",
        ],
        "search_spaces": {
            "outlier": [None, "clip"],  # , "zscore"
            "scale": [None, "standard"],  # , "robust"
            "select__method": ["lasso", "random_forest", "forward"],  # None, "forward", "variance",
            "select__k": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],  # , 20, 50, 100
            "reduce_pre__method": [None, "pca"],  # "pca", "umap", "tsne"
            "reduce_post__method": [None],  # "pca", "umap", "tsne"
            "clf": None,  # placeholder, to be filled with MODELS
        },
        # Metrics for which the best model should be persisted. The keys here
        # correspond to entries in ``scoring`` above, except for
        # ``maj_test_pct`` which represents the majority-vote baseline.
        "metrics": [
            "maj_test_pct",
            "accuracy",
            "balanced_accuracy",
            "neg_log_loss",
            "f1",
            "precision",
            "recall",
            "roc_auc",
        ],
        "scoring": {
            "accuracy": "accuracy",
            "balanced_accuracy": "balanced_accuracy",
            "neg_log_loss": "neg_log_loss",
            "f1": "f1",
            "precision": "precision",
            "recall": "recall",
            "roc_auc": "roc_auc",
        },
        "pipeline_order": [
            "outlier",
            "scale",
            "reduce_pre__method",
            "select__method",
            "select__k",
            "reduce_post__method",
            "clf",
        ],
        "run_otter_search_sex": {
            "n_iter": 30,
            "cv": "fold",
            "random_state": GLOBAL_RANDOM_SEED,
            "reuse_results": True,
        },
    },
    "pipeline_individual_id": {
        "pairwise_defaults": {
            "k_features": 16,
            "reducers": ["lda"],
            "selection_method": "forward",
            "n_components": 2,
            "outlier_methods": None,
            "scaler_methods": None,
            "use_sexmodel_prediction": False,
        },
        "search_spaces": {
            "outlier": [None],  # "clip", "zscore"
            "scale": [None, "standard"],  # None,, "robust"
            "select__method": ["forward", "random_forest", "lasso"],  # None,
            "select__k": [
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                9,
                10,
                11,
                12,
                13,
                14,
                15,
                16,
                17,
                18,
                19,
                20,
                21,
                22,
                23,
                24,
                25,
                26,
                27,
                28,
                29,
                30,
            ],
            "reduce__method": ["lda"],  # "pca", "umap",
            "n_components": [2],  # , 3, 4
            "use_sexmodel_prediction": [False, True],
        },
        "trail_generation_defaults": {
            "sample_size": 9,
            "subsample_sizes": [3, 5, 7],
            "n_candidates": 20,
        },
        "sequential_holdout_val_sizes": [2, 4, 6, 8],
    },
    "gui_annotator": {
        "default_scale": 1.0,
        # Default window size (w, h) in a widescreen ratio. The application
        # will automatically constrain this based on the available screen
        # resolution at runtime.
        "display_size": [1280, 720],
    },
    "visualisation": {
        "sex": {
            "value_map": SEX_VALUE_MAP,
            "colors": SEX_COLORS,
            "train_colors": TRAIN_COLORS,
            "test_colors": TEST_COLORS,
        }
    },
    "dataset_summary": {
        "species_labels": {
            "amur_tiger": {
                "common": "Amur Tiger",
                "latin": "Panthera tigris altaica",
            },
            "bengal_tiger": {
                "common": "Bengal Tiger",
                "latin": "Panthera tigris tigris",
            },
            "cheetah": {
                "common": "Cheetah",
                "latin": "Acinonyx jubatus",
            },
            "eurasian_otter": {
                "common": "Eurasian Otter",
                "latin": "Lutra lutra",
            },
            "giant_panda": {
                "common": "Giant Panda",
                "latin": "Ailuropoda melanoleuca",
            },
            "lowlandtapir": {
                "common": "Lowland Tapir",
                "latin": "Tapirus terrestris",
            },
            "mountain_lion": {
                "common": "Mountain Lion",
                "latin": "Puma concolor",
            },
            "white_rhino": {
                "common": "White Rhinoceros",
                "latin": "Ceratotherium simum",
            },
        }
    },
    "experiment": {"name": "dissertation_notebook"},
}

# Additional mappings for evaluation metrics used by the sex pipeline.
CONFIG["pipeline_sex"]["scoring_to_eval"] = {
    "accuracy": "accuracy_test",
    "balanced_accuracy": "balanced_test_acc",
    "neg_log_loss": "mean_test_neg_log_loss",
    "f1": "f1_test",
    "precision": "precision_test",
    "recall": "recall_test",
    "roc_auc": "roc_auc_test",
    "maj_test_pct": "maj_test_pct",
}
CONFIG["pipeline_sex"]["metric_map"] = {
    k: v
    for k, v in CONFIG["pipeline_sex"]["scoring_to_eval"].items()
    if k in CONFIG["pipeline_sex"]["scoring"] or k == "maj_test_pct"
}
