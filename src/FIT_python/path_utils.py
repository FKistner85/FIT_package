from pathlib import Path
from FIT_python.config import (
    PROCESSED_SPLITS_DIR,
    SCALED_DIR,
    FEATURE_SELECTED_DIR,
    DIM_REDUCED_DIR,
    normalize_dataset_name,
)


def split_path(dataset: str, split: str) -> Path:
    ds = normalize_dataset_name(dataset)
    return PROCESSED_SPLITS_DIR / f"{ds}_{split}.parquet"


def scaled_path(dataset: str, split: str) -> Path:
    ds = normalize_dataset_name(dataset)
    return SCALED_DIR / f"{ds}_{split}.parquet"


def feature_selected_path(dataset: str, split: str) -> Path:
    ds = normalize_dataset_name(dataset)
    return FEATURE_SELECTED_DIR / f"{ds}_{split}.parquet"


def dim_reduced_path(dataset: str, split: str) -> Path:
    ds = normalize_dataset_name(dataset)
    return DIM_REDUCED_DIR / f"{ds}_{split}.parquet"
