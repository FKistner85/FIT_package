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
        col_clean = re.sub(r'\W+', '_', col_clean)
        cleaned.append(col_clean)
    return cleaned

def load_csv(path: Path) -> pd.DataFrame:
    """Load a CSV file into a DataFrame."""
    return pd.read_csv(path)

def load_excel(path: Path) -> pd.DataFrame:
    """Load an Excel file into a DataFrame."""
    return pd.read_excel(path)

def load_raw_files(
    folder: Path,
    add_id: bool = True,
    id_prefix: Optional[str] = None
) -> Dict[str, pd.DataFrame]:
    """
    Load all CSV and Excel files from 'folder', clean columns, optionally add 'id' column.
    Returns a dict mapping file stem to DataFrame.
    """
    data_frames = {}
    for file in folder.iterdir():
        if file.suffix.lower() == '.csv':
            df = load_csv(file)
        elif file.suffix.lower() in ['.xlsx', '.xls']:
            df = load_excel(file)
        else:
            continue

        # Clean column names
        df.columns = clean_columns(df.columns)

        # Normalize individual identifier column
        if 'animal' in df.columns:
            df.rename(columns={'animal': 'individual_id'}, inplace=True)
        elif 'individual' in df.columns:
            df.rename(columns={'individual': 'individual_id'}, inplace=True)

        # Add id column if requested
        if add_id:
            stem = id_prefix if id_prefix else file.stem.replace(' ', '_')
            df.insert(0, 'id', [f"{stem}_{i}" for i in range(1, len(df) + 1)])

        # Attempt to coerce comma decimal strings to float
        df = coerce_numeric_columns(df)

        data_frames[file.stem.replace(' ', '_')] = df
    return data_frames


def coerce_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Convert numeric-looking object columns with comma decimal separator."""
    for col in df.select_dtypes(include="object").columns:
        series = df[col].astype(str).str.replace(',', '.', regex=False)
        numeric = pd.to_numeric(series, errors="coerce")
        # Convert if majority of non-null values are numeric
        if numeric.notna().sum() >= len(df) * 0.8 and numeric.notna().sum() > 0:
            df[col] = numeric.astype(float)
    return df

def sanitize_labels(
    df: pd.DataFrame,
    target_cols: List[str],
    skip_fillna: Optional[List[str]] = None,
    mapping: Optional[Dict[str, str]] = None
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
            df_clean[col] = df_clean[col].fillna('unknown')
        df_clean[col] = df_clean[col].astype(str).str.strip().str.lower()
        df_clean[col] = df_clean[col].str.replace(r"\W+", '_', regex=True)
        df_clean[col] = df_clean[col].map(mapping).fillna(df_clean[col])
    return df_clean
