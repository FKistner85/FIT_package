# src/FIT_python/data_import_wrapper.py

"""Wrapper for data import workflows, including label cleaning and numeric conversion."""

from pathlib import Path
import pandas as pd
from typing import Dict, Optional, List

from FIT_python.data_import_utils import load_raw_files, sanitize_labels
from FIT_python.transform_utils import convert_numeric
from FIT_python.config import DEFAULT_TARGETS, OTTER_META_COLS

class DataImporter:
    """Class to import, clean, and convert data from raw files."""

    def __init__(
        self,
        raw_dir: Path,
        target_cols: Optional[List[str]] = None,
        skip_fillna: Optional[List[str]] = None,
        label_map: Optional[Dict[str, str]] = None
    ):
        self.raw_dir = raw_dir
        self.target_cols = target_cols or []
        self.skip_fillna = skip_fillna or []
        self.label_map = label_map or {}

    def load(self) -> Dict[str, pd.DataFrame]:
        """Load raw files into DataFrames with cleaned columns and IDs."""
        return load_raw_files(self.raw_dir)

    def clean(self, dfs: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Apply label cleaning to each DataFrame's target columns."""
        cleaned = {}
        for name, df in dfs.items():
            df_clean = sanitize_labels(
                df,
                target_cols=self.target_cols,
                skip_fillna=self.skip_fillna,
                mapping=self.label_map
            )
            cleaned[name] = df_clean
        return cleaned

    def convert(self, dfs: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Convert feature columns to float (comma→dot) for numeric pipeline steps."""
        converted = {}
        for name, df in dfs.items():
            df_copy = df.copy()
            # determine feature columns: exclude meta and target columns
            if 'otter' in name.lower():
                meta_cols = OTTER_META_COLS
            else:
                meta_cols = ['id']
            feature_cols = [c for c in df_copy.columns if c not in meta_cols + self.target_cols]
            # convert these to numeric floats
            df_num = convert_numeric(df_copy, feature_cols)
            converted[name] = df_num
        return converted

    def run(self) -> Dict[str, pd.DataFrame]:
        """Load, clean labels, and convert numeric features in one step."""
        dfs = self.load()
        if self.target_cols:
            dfs = self.clean(dfs)
        # always convert numeric features before scaling
        dfs = self.convert(dfs)
        return dfs
