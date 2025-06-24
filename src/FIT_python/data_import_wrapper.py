"""Wrapper for data import workflows."""
from pathlib import Path
import pandas as pd
from typing import Dict, Optional, List

from FIT_python.data_import_utils import load_raw_files, sanitize_labels

class DataImporter:
    """Class to import and clean data from raw files."""
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

    def run(self) -> Dict[str, pd.DataFrame]:
        """Load and clean raw data in one step."""
        dfs = self.load()
        if self.target_cols:
            dfs = self.clean(dfs)
        return dfs
