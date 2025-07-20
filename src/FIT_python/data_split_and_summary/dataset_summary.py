"""Generate an overview table for all raw datasets."""

from __future__ import annotations

from pathlib import Path
import pandas as pd

from FIT_python.data_split_and_summary.data_import_wrapper import DataImporter
from FIT_python.data_split_and_summary.data_import_utils import get_feature_cols
from FIT_python.config import DEFAULT_TARGETS
from FIT_python.soft_config import SOFT_CONFIG


def _species_label(name: str, df: pd.DataFrame) -> str:
    """Return a human readable species label."""
    cfg = SOFT_CONFIG.get("dataset_summary", {}).get("species_labels", {})
    key = name.replace(" ", "_").lower()
    info = cfg.get(key)
    if info:
        return f"{info['common']} ({info['latin']})"

    common = name.replace("_", " ").capitalize()
    species_col = df.get("species") or df.get("Species")
    latin = None
    if species_col is not None and not species_col.dropna().empty:
        latin = str(species_col.dropna().iloc[0])
    latin = latin or common
    return f"{common} ({latin})"


def generate_dataset_overview(raw_dir: Path, out_csv: Path) -> pd.DataFrame:
    """Create a CSV summary of all raw datasets and return it as a DataFrame."""
    importer = DataImporter(raw_dir, target_cols=DEFAULT_TARGETS)
    dfs = importer.run()

    rows = []
    for name, df in dfs.items():
        feature_cols = get_feature_cols(df)
        row = {
            "Species": _species_label(name, df),
            "No. of footprints": len(df),
            "No. of known individuals": df.get("individual_id", pd.Series()).nunique(),
            "No. of trails": df.get("trail", pd.Series()).nunique(),
            "No. of measurements": len(feature_cols),
        }
        rows.append(row)

    overview_df = pd.DataFrame(rows)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    overview_df.to_csv(out_csv, index=False)
    return overview_df
