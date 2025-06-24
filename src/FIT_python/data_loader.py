from pathlib import Path

import pandas as pd
from typing import List, Dict, Optional
import re

# utils.py
def clean_columns(columns):
    """
    Clean column names: strip spaces, lowercase, replace non-alphanumeric with underscore.
    """
    cleaned = []
    for col in columns:
        col_clean = col.strip().lower()
        col_clean = re.sub(r'\W+', '_', col_clean)
        cleaned.append(col_clean)
    return cleaned

# data_loader.py
def load_csv(path: Path) -> pd.DataFrame:
    """
    Load a CSV file into a DataFrame.
    """
    return pd.read_csv(path)

def load_excel(path: Path) -> pd.DataFrame:
    """
    Load an Excel file into a DataFrame.
    """
    return pd.read_excel(path)

def load_raw_files(folder: Path) -> dict[str, pd.DataFrame]:
    """
    Loads all CSV and Excel files from the given folder,
    cleans column names, and adds an 'id' column.
    
    Returns:
        Dict mapping filename stem to DataFrame.
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

        # Standardize individual identifier column
        if "animal" in df.columns:
            df = df.rename(columns={"animal": "individual_id"})
        if "individual" in df.columns:
            df = df.rename(columns={"individual": "individual_id"})

        # Add id column
        stem = file.stem.replace(" ", "_")
        df.insert(0, 'id', [f"{stem}_{i}" for i in range(1, len(df)+1)])

        data_frames[stem] = df

    return data_frames



def sanitize_labels(
    df: pd.DataFrame,
    target_cols: List[str],
    skip_fillna: Optional[List[str]] = None,
    mapping: Optional[Dict[str, str]] = None
) -> pd.DataFrame:
    """
    Cleans string labels in target_cols:
      - For cols not in skip_fillna: fillna("unknown")
      - strip() & lower()
      - replace non-word chars with underscore
      - apply a mapping dict (e.g. 'f'->'female', 'm'->'male')
    """
    df_clean = df.copy()
    skip_fillna = skip_fillna or []
    mapping = mapping or {}

    for col in target_cols:
        # 1) fillna except for columns in skip_fillna
        if col not in skip_fillna:
            df_clean[col] = df_clean[col].fillna("unknown")

        # 2) lowercase & trim
        df_clean[col] = df_clean[col].astype(str).str.strip().str.lower()

        # 3) replace non-alphanumeric with underscore
        df_clean[col] = df_clean[col].str.replace(r"\W+", "_", regex=True)

        # 4) apply mapping dict and keep original if not in mapping
        df_clean[col] = df_clean[col].map(mapping).fillna(df_clean[col])

    return df_clean


