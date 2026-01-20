"""Central configuration parameters and file paths used throughout the package."""

from pathlib import Path
import os
from datetime import datetime
import json
import hashlib

# ============================================================================
# Paths
# ============================================================================
_root_env = os.getenv("FIT_EXPERIMENT_ROOT")
if _root_env is not None:
    EXPERIMENT_ROOT = Path(_root_env)
else:
    exp_name = os.getenv("FIT_EXPERIMENT_NAME") or datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    EXPERIMENT_ROOT = Path.cwd() / "results" / exp_name

EXPERIMENT_ROOT.mkdir(parents=True, exist_ok=True)

_raw_env = os.getenv("FIT_RAW_DIR")
if _raw_env is not None:
    RAW_DIR = Path(_raw_env)
else:
    RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
    if not RAW_DIR.exists():
        raise FileNotFoundError(f"RAW_DIR not found: {RAW_DIR}")

DATA_DIR = EXPERIMENT_ROOT / "data"; DATA_DIR.mkdir(parents=True, exist_ok=True)
CLEANED_DIR = DATA_DIR / "cleaned"
SPLITS_DIR = DATA_DIR / "splits"; SPLITS_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR = DATA_DIR / "processed"
PROCESSED_SPLITS_DIR = PROCESSED_DIR / "splits"
SCALED_DIR = PROCESSED_DIR / "scaled"
FEATURE_SELECTED_DIR = PROCESSED_DIR / "feature_selected"
DIM_REDUCED_DIR = PROCESSED_DIR / "dim_reduced"
NUMERIC_DIR = PROCESSED_DIR / "numeric"

# unified results directory with dedicated subfolders
# NOTE: EXPERIMENT_ROOT already points to the experiment-specific directory (e.g., "results/fit_notebook_final")
# so we don't need an additional "results" subdirectory to avoid double nesting like "results/results/"
RESULTS_DIR = EXPERIMENT_ROOT
RESULTS_SUBDIRS = {
    name: RESULTS_DIR / name for name in ["dataprocessing", "sex_modelling", "individual_id"]
}
for path in RESULTS_SUBDIRS.values():
    path.mkdir(parents=True, exist_ok=True)
    (path / "figures").mkdir(parents=True, exist_ok=True)
    (path / "tables").mkdir(parents=True, exist_ok=True)

OTTER_LANDMARK_MAP_PATH = PROCESSED_DIR / "otter_landmark_map.json"
OTTER_POINT_MAP_PATH = PROCESSED_DIR / "otter_point_map.json"

PATHS = {
    "experiment_root": EXPERIMENT_ROOT,
    "raw_data": RAW_DIR,
    "data": DATA_DIR,
    "cleaned": CLEANED_DIR,
    "splits": SPLITS_DIR,
    "processed": PROCESSED_DIR,
    "processed_splits": PROCESSED_SPLITS_DIR,
    "scaled": SCALED_DIR,
    "feature_selected": FEATURE_SELECTED_DIR,
    "dim_reduced": DIM_REDUCED_DIR,
    "numeric": NUMERIC_DIR,
    "results": RESULTS_DIR,
    "dataprocessing": RESULTS_SUBDIRS["dataprocessing"],
    "sex_modelling": RESULTS_SUBDIRS["sex_modelling"],
    "individual_id": RESULTS_SUBDIRS["individual_id"],
    "otter_landmark_map": OTTER_LANDMARK_MAP_PATH,
    "otter_point_map": OTTER_POINT_MAP_PATH,
}

# ============================================================================
# Default parameters, mappings & visualisation
# ============================================================================
OTTER_META_COLS = ["id", "date", "location", "dataorigin", "substrate"]
DEFAULT_TARGETS = ["species", "individual_id", "trail", "sex"]


def _lighten(color: str, amount: float) -> str:
    color = color.lstrip("#")
    r = int(color[0:2], 16) / 255
    g = int(color[2:4], 16) / 255
    b = int(color[4:6], 16) / 255
    r = round((r + (1 - r) * amount) * 255)
    g = round((g + (1 - g) * amount) * 255)
    b = round((b + (1 - b) * amount) * 255)
    return f"#{r:02x}{g:02x}{b:02x}"


SEX_VALUE_MAP = {"f": "Female", "F": "Female", 0: "Female", "m": "Male", "M": "Male", 1: "Male"}
SEX_COLORS = {"Female": "#800000", "Male": "#000080", "Unknown": "#FFA500"}
TRAIN_COLORS = SEX_COLORS
TEST_COLORS = {k: _lighten(v, 0.5) for k, v in SEX_COLORS.items()}

DATASET_COLOR_SHADES = {
    "Own Data Collection": 0.0,
    "Vetrecova et al": 0.2,
    "Fieldprints Lower Saxony": 0.4,
}

GROUP_COL = "individual_id"
STRATIFY_COL = "sex"
GLOBAL_RANDOM_SEED =123
TEST_SIZE = 0.3
NUM_FOLDS = 5
DEBUG_MODE = False

DEFAULT_METADATA_COLS = [
    "individual_id",
    "Trail",
    "sex",
    "id",
    "Fold",
]

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

# create species specific figures and tables folders
for sub in RESULTS_SUBDIRS.values():
    for species in SPECIES_MODEL_MAP.values():
        (sub / species / "figures").mkdir(parents=True, exist_ok=True)
        (sub / species / "tables").mkdir(parents=True, exist_ok=True)

CONFIG = {
    "data_split_and_summary": {
        "sex_categories": ["Female", "Male"],
        "split_labels": ["Train", "Test"],
        "species_remap": {
            "a_j_soemmeringii": "acinonyx_jubatus_soemmeringii",
            "a_j_jubatus": "acinonyx_jubatus_jubatus",
        },
        "generate_histograms": False,
        "reuse_overview": False,
        "reuse_splits": True,
        "fold_column_name": "Fold",
        "trail_column_name": "Trail",
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
            "supervised": False,
        },
        "metadata_cols": ["individual_id", "Trail", "sex", "id", "Fold"],
        "n_jobs": -1,
    },
    "gui_annotator": {"default_scale": 1.0, "display_size": [1280, 720]},
    "visualisation": {
        "sex": {
            "value_map": SEX_VALUE_MAP,
            "colors": SEX_COLORS,
            "train_colors": TRAIN_COLORS,
            "test_colors": TEST_COLORS,
        },
        "dataset": {"color_shades": DATASET_COLOR_SHADES},
        "dpi": 300,
        "figure_sizes": {
            "umap_plot": (12, 8),
            "benchmark_bar": (14, 7),
            "histogram": (10, 8),
            "cutoff_plot": (10, 6),
            "dendrogram": (14, 6),
            "scatter_reg": (7, 7),
        },
        "font_sizes": {
            "axis_label": 12,
            "legend": 10,
            "title": 14,
            "tick": 11,
            "panel_label": 12,
        },
        "error_bar_capsize": 3,
        "histogram_bins": 10,
        "top_k_pipelines": 3,
    },
    "dataset_summary": {
        "species_labels": {
            "amur_tiger": {"common": "Amur Tiger", "latin": "Panthera tigris altaica"},
           "bengal_tiger": {"common": "Bengal Tiger", "latin": "Panthera tigris tigris"},
            "cheetah": {"common": "Cheetah", "latin": "Acinonyx jubatus"},
            "eurasian_otter": {"common": "Eurasian Otter", "latin": "Lutra lutra"},
            "giant_panda": {"common": "Giant Panda", "latin": "Ailuropoda melanoleuca"},
            "lowlandtapir": {"common": "Lowland Tapir", "latin": "Tapirus terrestris"},
            "mountain_lion": {"common": "Mountain Lion", "latin": "Puma concolor"},
            "white_rhino": {"common": "White Rhinoceros", "latin": "Ceratotherium simum"},
        }
    },
    "experiment": {"name": "dissertation_notebook"},
}

# ============================================================================
# Sex modeling
# ============================================================================
SEXMODEL_SETTINGS = {
    "sex_model": "bayes_search",
    "metric_key": "balanced_accuracy",
}
BAYES_REFIT = "balanced_accuracy"
SEX_PREDICT_METRIC = "balanced_accuracy"

# === CONFIG["pipeline_sex"] verschlankt auf 4 Typen ===
# in FIT_python/config.py (oder wo du deinen CONFIG Block definierst)
CONFIG["pipeline_sex"] = {
    "model_keys": [
        # nur robuste LDA + XGB
        "lda_svd",
        "xgb_std",
    ],
    "search_spaces": {
        # globale Pipeline-Schritte
        "outlier": [None, "clip"],
        "scale": [None, "standard"],
        "select__method": ["forward", "lasso", "random_forest"],
        "select__k": list(range(2, 31)),
        "reduce_pre__method": [None, "pca"],
        "reduce_pre__n_components": [None, 2, 3, 5, 8, 12],  # mehr Komponenten
        "reduce_post__method": [None],

        # modell-spezifische Hyperparameter
        "clf": [
            # LDA (nur svd)
            ("lda_svd", {"clf__n_components": [1]}),  # shrinkage nicht erlaubt bei svd

            # XGBoost (reguliert)
            ("xgb_std", {
                "clf__n_estimators":     [100, 200, 300],
                "clf__max_depth":        [3, 4, 5, 6, 8],
                "clf__learning_rate":    [0.01, 0.03, 0.1],
                "clf__subsample":        [0.7, 0.8, 0.9, 1.0],
                "clf__colsample_bytree": [0.6, 0.8, 1.0],
                "clf__min_child_weight": [1, 3, 5],
                "clf__reg_alpha":        [0.0, 0.5, 1.0],
                "clf__reg_lambda":       [1.0, 2.0, 4.0],
                "clf__gamma":            [0, 1, 5],
            }),
        ],
    },

    "metrics": [
        "accuracy",
        "balanced_accuracy",
        "neg_log_loss",
        "f1",
        "precision",
        "recall",
        "roc_auc",
    ],

    # Multi-scoring bleibt; refit konfigurierst du separat über BAYES_REFIT
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
        "reduce_pre__n_components",
        "select__method",
        "select__k",
        "reduce_post__method",
        "clf",
    ],

    "run_otter_search_sex": {
        "n_iter": 50,
        "cv": "fold",
        "random_state": GLOBAL_RANDOM_SEED,
        "reuse_results": True,
    },

    # Metriken jetzt auf CV-Resultate gemappt
    "scoring_to_eval": {
        "accuracy": "mean_cv_accuracy",
        "balanced_accuracy": "mean_cv_balanced_accuracy",
        "neg_log_loss": "mean_cv_neg_log_loss",
        "f1": "mean_cv_f1",
        "precision": "mean_cv_precision",
        "recall": "mean_cv_recall",
        "roc_auc": "mean_cv_roc_auc",
    },
}

# metric_map identisch zu scoring_to_eval
CONFIG["pipeline_sex"]["metric_map"] = CONFIG["pipeline_sex"]["scoring_to_eval"].copy()

PIPE_SEARCH = CONFIG["pipeline_sex"]["search_spaces"]


# ============================================================================
# Individual id modeling
# ============================================================================
CONFIG["pipeline_individual_id"] = {
    "pairwise_defaults": {
        "k_features": 16,
        "reducers": ["lda"],
        "selection_method": "forward",
        "n_components": 2,
        "outlier_methods": None,
        "scaler_methods": None,
        "use_sexmodel_prediction": False,
        "reuse_summary": False,
        "overwrite_results": False,
    },
    "search_spaces": {
        "outlier": [None],
        "scale": ["standard"],
        "select__method": ["forward", "random_forest", "lasso"],
        "select__k": list(range(2, 31, 2)),  # hier statt '=' ein ':'
        "reduce__method": ["lda"],
        "n_components": [2],
        "use_sexmodel_prediction": [False, True],
    },
    "trail_generation_defaults": {
        "sample_size": 9,
        "subsample_sizes": [3, 5, 7],
        "n_candidates": 20,
    },
    "clustering": {
        "alpha": 0.5,
        "cutoff_range": {"start": 0.4, "stop": 4.2, "step": 0.1},
        "distance_metrics": ["euclidean", "manhattan", "cosine", "chebyshev", "canberra", "braycurtis"],
    },
    "sequential_holdout": {
        "iterations": 30,
        "val_sizes_train": list(range(2, 21)),
        "val_sizes_test": [2, 4, 6, 8],
    },
    "sequential_holdout_val_sizes": [2, 4, 6, 8],
}


# ============================================================================
# Config Logging System
# ============================================================================
def _serialize_config() -> dict:
    """Serialize the current CONFIG to a JSON-compatible dict."""
    def _convert(obj):
        if isinstance(obj, Path):
            return str(obj)
        elif isinstance(obj, dict):
            return {str(k): _convert(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [_convert(item) for item in obj]
        elif isinstance(obj, (int, float, str, bool, type(None))):
            return obj
        else:
            # For any other type, convert to string
            return str(obj)
    
    return {
        "timestamp": datetime.now().isoformat(),
        "experiment_name": os.getenv("FIT_EXPERIMENT_NAME", "default"),
        "experiment_root": str(EXPERIMENT_ROOT),
        "raw_dir": str(RAW_DIR),
        "global_random_seed": GLOBAL_RANDOM_SEED,
        "num_folds": NUM_FOLDS,
        "debug_mode": DEBUG_MODE,
        "config": _convert(CONFIG),
        "paths": _convert(PATHS),
    }


def _compute_config_hash(config_dict: dict) -> str:
    """Compute a hash of the config to detect changes."""
    # Exclude timestamp from hash computation
    config_copy = config_dict.copy()
    config_copy.pop("timestamp", None)
    config_str = json.dumps(config_copy, sort_keys=True)
    return hashlib.md5(config_str.encode()).hexdigest()[:8]


def save_config_snapshot(reason: str = "manual") -> Path:
    """Save a snapshot of the current configuration.
    
    Parameters
    ----------
    reason : str
        Reason for the snapshot (e.g., "initial", "manual", "modified")
        
    Returns
    -------
    Path
        Path to the saved config file
    """
    config_dict = _serialize_config()
    config_hash = _compute_config_hash(config_dict)
    
    config_dir = EXPERIMENT_ROOT / "config_logs"
    config_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"config_{timestamp}_{config_hash}_{reason}.json"
    filepath = config_dir / filename
    
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(config_dict, f, indent=2)
    
    # Also maintain a "latest" symlink/copy
    latest_path = config_dir / "config_latest.json"
    with open(latest_path, "w", encoding="utf-8") as f:
        json.dump(config_dict, f, indent=2)
    
    return filepath


def _auto_log_config():
    """Automatically log config on import if not already logged."""
    config_dir = EXPERIMENT_ROOT / "config_logs"
    latest_path = config_dir / "config_latest.json"
    
    current_config = _serialize_config()
    current_hash = _compute_config_hash(current_config)
    
    should_save = True
    reason = "initial"
    
    if latest_path.exists():
        with open(latest_path, "r", encoding="utf-8") as f:
            previous_config = json.load(f)
        previous_hash = _compute_config_hash(previous_config)
        
        if current_hash == previous_hash:
            should_save = False  # Config hasn't changed
        else:
            reason = "modified"
    
    if should_save:
        filepath = save_config_snapshot(reason=reason)
        if reason == "initial":
            print(f"[CONFIG] Initial config saved to: {filepath.relative_to(EXPERIMENT_ROOT)}")
        elif reason == "modified":
            print(f"[CONFIG] Config changes detected and saved to: {filepath.relative_to(EXPERIMENT_ROOT)}")


# Auto-log config on module import
_auto_log_config()


# ============================================================================
# Public API for manual config logging
# ============================================================================
def log_config_change(reason: str = "manual_update"):
    """Manually log a config snapshot after making changes.
    
    Use this in notebooks after modifying CONFIG parameters to track changes.
    
    Parameters
    ----------
    reason : str
        Description of what was changed (e.g., "increased_num_folds", "changed_random_seed")
        
    Example
    -------
    >>> import FIT_python.config as cfg
    >>> cfg.CONFIG["pipeline_individual_id"]["trail_generation_defaults"]["sample_size"] = 12
    >>> cfg.log_config_change("changed_trail_sample_size_to_12")
    """
    filepath = save_config_snapshot(reason=reason)
    print(f"[CONFIG] Config snapshot saved: {filepath.relative_to(EXPERIMENT_ROOT)}")
    return filepath
