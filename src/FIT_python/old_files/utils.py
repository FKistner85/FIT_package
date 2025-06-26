# src/FIT_python/transform_utils.py

import pandas as pd
import numpy as np
from pathlib import Path
import json
from typing import Tuple, Dict, List

def convert_numeric(
    df: pd.DataFrame,
    feature_cols: List[str]
) -> pd.DataFrame:
    """Convert feature columns to float, replacing comma decimal separators.

    Non-convertible values are coerced to NaN so that mixed columns do not
    raise errors during conversion.
    """
    for col in feature_cols:
        df[col] = (
            df[col]
            .astype(str)
            .str.replace(',', '.', regex=False)
            .pipe(pd.to_numeric, errors="coerce")
        )
    return df

def one_hot_encode_targets(
    df: pd.DataFrame,
    target_cols: List[str]
) -> Tuple[np.ndarray, Dict[str, List[str]]]:
    """
    One-hot encode target columns.
    Returns:
      - y: numpy array of shape (n_samples, total_classes)
      - mapping: dict target_col -> list of classes
    """
    mappings: Dict[str, List[str]] = {}
    y_frames = []
    for col in target_cols:
        dummies = pd.get_dummies(df[col], prefix=col)
        mappings[col] = list(dummies.columns)
        y_frames.append(dummies)
    y_df = pd.concat(y_frames, axis=1)
    return y_df.to_numpy(), mappings

def save_target_mapping(
    mapping: Dict[str, List[str]],
    path: Path
) -> None:
    """Save target mapping dict to JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(mapping, f, indent=2)
