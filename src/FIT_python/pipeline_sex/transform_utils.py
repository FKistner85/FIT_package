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
    """Convert feature columns to numeric and encode categorical strings.

    Steps
    -----
    1. Replace comma decimal separators with dots.
    2. Attempt numeric conversion via ``pd.to_numeric``.
    3. Columns that contain only non-numeric values are one-hot encoded and the
       original column dropped.

    Returns the modified ``DataFrame`` containing only numeric columns.
    """

    categorical_cols: list[str] = []
    for col in feature_cols:
        ser = df[col].astype(str).str.replace(',', '.', regex=False)
        numeric = pd.to_numeric(ser, errors="coerce")

        # if all values are NaN after conversion, treat as categorical
        if numeric.notna().sum() == 0:
            df[col] = ser
            categorical_cols.append(col)
        else:
            df[col] = numeric

    if categorical_cols:
        dummies = pd.get_dummies(df[categorical_cols], prefix=categorical_cols)
        df = pd.concat([df.drop(columns=categorical_cols), dummies], axis=1)

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
