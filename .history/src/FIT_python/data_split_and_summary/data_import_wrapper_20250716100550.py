# src/FIT_python/data_import_wrapper.py

"""Wrapper for data import workflows, including label cleaning and numeric conversion."""

from pathlib import Path
import pandas as pd
from typing import Dict, Optional, List

from FIT_python.data_split_and_summary.data_import_utils import (
    load_raw_files,
    sanitize_labels,
)
from FIT_python.data_split_and_summary.transform_utils import convert_numeric
import FIT_python.config as config
from FIT_python.config import DEFAULT_TARGETS, OTTER_META_COLS
import sys


class DataImporter:
    """Class to import, clean, and convert data from raw files."""

    def __init__(
        self,
        raw_dir: Path,
        target_cols: Optional[List[str]] = None,
        skip_fillna: Optional[List[str]] = None,
        label_map: Optional[Dict[str, str]] = None,
    ):
        self.raw_dir = raw_dir
        self.target_cols = target_cols or []
        self.skip_fillna = skip_fillna or []
        self.label_map = label_map or {}
        self._next_id = 0

    def load(self) -> Dict[str, pd.DataFrame]:
        """Load raw files into DataFrames with cleaned columns and IDs."""
        dfs = load_raw_files(self.raw_dir, add_id=False)
        for name, df in dfs.items():
            n = len(df)
            # inserting repeatedly can lead to fragmentation; assign instead
            df = df.copy()
            df["id"] = range(self._next_id, self._next_id + n)
            # ensure id is the first column
            cols = ["id"] + [c for c in df.columns if c != "id"]
            dfs[name] = df.loc[:, cols]
            if config.DEBUG_MODE:
                print(f"[DEBUG] loaded {name}: shape={df.shape}")
            self._next_id += n
        return dfs

    def clean(self, dfs: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Apply label cleaning to each DataFrame's target columns."""
        cleaned = {}
        for name, df in dfs.items():
            df_clean = sanitize_labels(
                df,
                target_cols=self.target_cols,
                skip_fillna=self.skip_fillna,
                mapping=self.label_map,
            )
            cleaned[name] = df_clean
            if config.DEBUG_MODE:
                print(f"[DEBUG] cleaned {name}: shape={df_clean.shape}")
        return cleaned

    def convert(self, dfs: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Convert feature columns to float (comma→dot) for numeric pipeline steps."""
        converted = {}
        for name, df in dfs.items():
            df_copy = df.copy()
            # determine feature columns: exclude meta and target columns
            if "otter" in name.lower():
                meta_cols = OTTER_META_COLS
            else:
                meta_cols = ["id"]
            feature_cols = [
                c for c in df_copy.columns if c not in meta_cols + self.target_cols
            ]
            # convert these to numeric floats
            df_num = convert_numeric(df_copy, feature_cols)
            converted[name] = df_num
            if config.DEBUG_MODE:
                print(f"[DEBUG] converted {name}: shape={df_num.shape}")
        return converted

    def run(self) -> Dict[str, pd.DataFrame]:
        """Load, clean labels, and convert numeric features in one step."""
        dfs = self.load()
        if self.target_cols:
            dfs = self.clean(dfs)
        # always convert numeric features before scaling
        dfs = self.convert(dfs)
        return dfs


class DataImportWrapper:
    """High-level wrapper to clean all raw datasets and persist Parquet files."""

    def __init__(self) -> None:
        pass

    def clean_all(self) -> int:
        """Load raw files, clean labels and save Parquet outputs.

        Returns 0 on success, 1 on failure.
        """
        if not config.RAW_DIR.exists():
            msg = f"Required directory not found: {config.RAW_DIR}"
            print(f"[ERROR] {msg}", file=sys.stderr)
            if config.DEBUG_MODE:
                raise FileNotFoundError(msg)
            return 1

        config.CLEANED_DIR.mkdir(parents=True, exist_ok=True)
        importer = DataImporter(config.RAW_DIR, target_cols=DEFAULT_TARGETS)

        # load raw and keep copy for plotting
        raw_dfs = importer.load()
        cleaned_dfs = importer.clean(raw_dfs)
        dfs = importer.convert(cleaned_dfs)
        if config.DEBUG_MODE:
            print("[DEBUG] saving cleaned datasets")

        for name, df in dfs.items():
            out = config.CLEANED_DIR / f"{name}.parquet"
            df.to_parquet(out, index=False)
            print(f"[REPORT] cleaned {name}: shape={df.shape}")

            raw_df = raw_dfs.get(name)
            if raw_df is not None:
                fig_dir = config.FIGURES_DIR / "feature_distributions" / name
                try:
                    from FIT_python.general_pipeline_steps.outlier_wrapper2 import (
                        plot_feature_distributions,
                    )

                    plot_feature_distributions(raw_df, df, fig_dir)
                except Exception as exc:
                    print(f"[WARN] plotting failed for {name}: {exc}")

                # correlation heatmap of feature groups
                try:
                    from FIT_python.Visualisations.feature_corr_utils import (
                        plot_feature_correlations,
                    )

                    corr_dir = config.FIGURES_DIR / "feature_correlations" / name
                    plot_feature_correlations(df, corr_dir)
                except Exception as exc:
                    print(f"[WARN] correlation plot failed for {name}: {exc}")
        print("[SUCCESS] clean_all completed.")
        return 0
