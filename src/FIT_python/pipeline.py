# src/FIT_python/pipeline.py

from pathlib import Path
import pandas as pd
from typing import Dict, Any, List, Optional

from FIT_python.data_loader import load_raw_files, load_csv, load_excel, sanitize_labels

class DataPipeline:
    def __init__(
        self,
        raw_dir: Optional[Path] = None,
        raw_file: Optional[Path] = None,
        col_mapping: Dict[str, Dict[str, List[str]]] = {},
        skip_fillna: List[str] = [],
        label_map: Dict[str, str] = {}
    ):
        """
        Args:
            raw_dir:      Path to folder with many raw files. Mutually exclusive with raw_file.
            raw_file:     Path to a single raw file. Mutually exclusive with raw_dir.
            col_mapping:  Mapping dataset-name → {'meta': [...], 'targets': [...]}.
            skip_fillna:  List of target columns to skip fillna on.
            label_map:    Mapping raw label → cleaned label (e.g. 'f'->'female').
        """
        if (raw_dir is None) == (raw_file is None):
            raise ValueError("You must specify exactly one of raw_dir or raw_file")
        self.raw_dir = raw_dir
        self.raw_file = raw_file
        self.col_mapping = col_mapping
        self.skip_fillna = skip_fillna
        self.label_map = label_map

    def load(self) -> Dict[str, pd.DataFrame]:
        """Load raw data: either all files in a directory or a single file."""
        if self.raw_dir:
            return load_raw_files(self.raw_dir)
        else:
            # load single file and wrap in a dict keyed by stem
            path = self.raw_file
            if path.suffix.lower() == '.csv':
                df = load_csv(path)
            elif path.suffix.lower() in ['.xlsx', '.xls']:
                df = load_excel(path)
            else:
                raise ValueError(f"Unsupported file type: {path.suffix}")

            # clean only the column names
            from FIT_python.data_loader import clean_columns
            df.columns = clean_columns(df.columns)

            # add id column
            stem = path.stem.replace(" ", "_")
            df.insert(0, 'id', [f"{stem}_{i}" for i in range(1, len(df) + 1)])

            return {stem: df}


    def clean_targets(self, df: pd.DataFrame, target_cols: List[str]) -> pd.DataFrame:
        """Sanitize only the target columns in the DataFrame."""
        return sanitize_labels(
            df,
            target_cols=target_cols,
            skip_fillna=self.skip_fillna,
            mapping=self.label_map
        )

    def split(
        self,
        df: pd.DataFrame,
        meta_cols: List[str],
        target_cols: List[str]
    ) -> Dict[str, Any]:
        """
        Split df into:
          - 'meta': DataFrame with only meta_cols
          - 'target': DataFrame with only target_cols (or None)
          - 'features': DataFrame with remaining feature columns
          - 'feature_cols': list of those feature column names
        """
        meta_df = df[meta_cols]
        target_df = df[target_cols] if target_cols else None
        feature_cols = [c for c in df.columns if c not in meta_cols + target_cols]
        features_df = df[feature_cols]
        return {
            'meta': meta_df,
            'target': target_df,
            'features': features_df,
            'feature_cols': feature_cols
        }

    def run(self) -> Dict[str, Dict[str, Any]]:
        """Run the full pipeline: load, clean targets, split."""
        results: Dict[str, Dict[str, Any]] = {}
        raw_dfs = self.load()

        for name, df in raw_dfs.items():
            mapping = self.col_mapping[name]
            meta_cols = mapping['meta']
            target_cols = mapping['targets']

            # Clean target columns
            df_clean_targets = self.clean_targets(df, target_cols)
            df[target_cols] = df_clean_targets[target_cols]

            # Split into meta, target, features
            split_parts = self.split(df, meta_cols, target_cols)
            results[name] = split_parts

        return results
