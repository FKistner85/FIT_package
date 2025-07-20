"""Utility functions for data import and cleaning."""

import re
from pathlib import Path
import pandas as pd
from typing import List, Dict, Optional, Union


def clean_columns(columns: Union[List[str], pd.Index]) -> List[str]:
    """
    Clean column names: strip spaces, lowercase, replace non-alphanumeric with underscore.
    """
    cleaned = []
    for col in columns:
        col_clean = col.strip().lower()
        col_clean = re.sub(r"\W+", "_", col_clean)
        cleaned.append(col_clean)
    return cleaned


def load_csv(path: Path) -> pd.DataFrame:
    """Load a CSV file into a DataFrame."""
    return pd.read_csv(path)


def load_excel(path: Path) -> pd.DataFrame:
    """Load an Excel file into a DataFrame."""
    return pd.read_excel(path)


def load_raw_files(
    folder: Path, add_id: bool = True, id_prefix: Optional[str] = None
) -> Dict[str, pd.DataFrame]:
    """
    Load all CSV and Excel files from 'folder', clean columns, optionally add 'id' column.
    Returns a dict mapping file stem to DataFrame.
    """
    data_frames = {}
    for file in folder.iterdir():
        if file.suffix.lower() == ".csv":
            df = load_csv(file)
        elif file.suffix.lower() in [".xlsx", ".xls"]:
            df = load_excel(file)
        else:
            continue

        # Clean column names
        df.columns = clean_columns(df.columns)

        # Normalize individual identifier column
        if "animal" in df.columns:
            df.rename(columns={"animal": "individual_id"}, inplace=True)
        elif "individual" in df.columns:
            df.rename(columns={"individual": "individual_id"}, inplace=True)

        # Add id column if requested
        if add_id:
            stem = id_prefix if id_prefix else file.stem.replace(" ", "_")
            df.insert(0, "id", [f"{stem}_{i}" for i in range(1, len(df) + 1)])

        # Attempt to coerce comma decimal strings to float
        df = coerce_numeric_columns(df)

        data_frames[file.stem.replace(" ", "_")] = df
    return data_frames


def coerce_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Convert numeric-looking object columns with comma decimal separator."""
    for col in df.select_dtypes(include="object").columns:
        series = df[col].astype(str).str.replace(",", ".", regex=False)
        numeric = pd.to_numeric(series, errors="coerce")
        # Convert if majority of non-null values are numeric
        if numeric.notna().sum() >= len(df) * 0.8 and numeric.notna().sum() > 0:
            df[col] = numeric.astype(float)
    return df


def sanitize_labels(
    df: pd.DataFrame,
    target_cols: List[str],
    skip_fillna: Optional[List[str]] = None,
    mapping: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    """
    Clean string labels in target_cols:
      - fillna('unknown') except skip_fillna
      - strip & lower
      - replace non-word with underscore
      - apply mapping dict
    Returns a new DataFrame.
    """
    df_clean = df.copy()
    skip_fillna = skip_fillna or []
    mapping = mapping or {}

    for col in target_cols:
        if col not in skip_fillna:
            df_clean[col] = df_clean[col].fillna("unknown")
        df_clean[col] = df_clean[col].astype(str).str.strip().str.lower()
        df_clean[col] = df_clean[col].str.replace(r"\W+", "_", regex=True)
        df_clean[col] = df_clean[col].map(mapping).fillna(df_clean[col])
    return df_clean


def load_and_prep_df_individual(species: str, splits_dir: Path) -> pd.DataFrame:
    """
    Lädt das Train-Parquet für eine Spezies, filtert nur Männchen/Weibchen
    und entfernt die Fold-Spalte, falls vorhanden.
    """
    fp = splits_dir / species / "train.parquet"
    df = pd.read_parquet(fp)
    df = df.drop(columns=["Fold"], errors="ignore")
    return df


def get_feature_cols(df: pd.DataFrame) -> list[str]:
    """Return the numeric feature columns of ``df``.

    Feature columns are detected either by common prefixes or, if no
    such columns exist, by selecting all numeric columns starting from
    column index 5.
    """

    prefixes = ("v", "area", "dist", "ang", "t")
    has_prefix_cols = any(col.lower().startswith(prefixes) for col in df.columns)

    if has_prefix_cols:
        return [
            c
            for c in df.columns
            if c.lower().startswith(prefixes) and pd.api.types.is_numeric_dtype(df[c])
        ]
    return [c for c in df.columns[5:] if pd.api.types.is_numeric_dtype(df[c])]


import pandas as pd
import numpy as np
from pathlib import Path
import json
from typing import Tuple, Dict, List


def convert_numeric(df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    """Convert feature columns to float, replacing comma decimal separators.

    Non-convertible values are coerced to NaN so that mixed columns do not
    raise errors during conversion.

    This implementation performs the conversion on all columns at once which
    avoids DataFrame fragmentation warnings when many columns are processed.
    """

    if not feature_cols:
        return df

    ser = (
        df[feature_cols]
        .astype(str)
        .replace({",": "."}, regex=False)
        .apply(pd.to_numeric, errors="coerce")
    )
    df[feature_cols] = ser
    return df


def one_hot_encode_targets(
    df: pd.DataFrame, target_cols: List[str]
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


def save_target_mapping(mapping: Dict[str, List[str]], path: Path) -> None:
    """Save target mapping dict to JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(mapping, f, indent=2)
