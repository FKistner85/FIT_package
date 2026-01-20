"""Generate an overview table for all raw datasets."""

from __future__ import annotations

from pathlib import Path
import pandas as pd

from FIT_python.data_split_and_summary.data_import_wrapper import DataImporter
from FIT_python.data_split_and_summary.data_import_utils import get_feature_cols
from tqdm.auto import tqdm
from FIT_python.config import DEFAULT_TARGETS
from FIT_python.config import CONFIG


def _species_label(name: str, df: pd.DataFrame) -> str:
    """Return a human readable species label."""
    cfg = CONFIG.get("dataset_summary", {}).get("species_labels", {})
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


def generate_dataset_overview(
    raw_dir: Path, out_path: Path, reuse: bool = True
) -> pd.DataFrame:
    """Create a Parquet summary of all raw datasets.

    Parameters
    ----------
    raw_dir:
        Directory containing the raw footprint tables.
    out_path:
        Destination of the generated overview table.
    reuse:
        When ``True`` and ``out_path`` already exists, the Parquet file is loaded
        and returned instead of recomputing the statistics.
    """

    if reuse and out_path.exists():
        return pd.read_parquet(out_path)

    importer = DataImporter(raw_dir, target_cols=DEFAULT_TARGETS)
    dfs = importer.run()

    rows = []
    for name, df in tqdm(dfs.items(), desc="Datasets"):
        feature_cols = get_feature_cols(df)
        
        # Count individuals by sex
        sex_col = df.get("sex", pd.Series())
        individual_id_col = df.get("individual_id", pd.Series())
        
        # Count unique individuals per sex category
        n_female = 0
        n_male = 0
        if not sex_col.empty and not individual_id_col.empty:
            # Create a df with individual_id and sex, drop duplicates to count unique individuals
            ind_sex = df[["individual_id", "sex"]].dropna().drop_duplicates(subset="individual_id")
            # Normalize sex values to handle different formats (f/F/0/Female, m/M/1/Male)
            ind_sex_norm = ind_sex["sex"].astype(str).str.lower().str.strip()
            n_female = ind_sex_norm.isin(["f", "female", "0"]).sum()
            n_male = ind_sex_norm.isin(["m", "male", "1"]).sum()
        
        row = {
            "Species": _species_label(name, df),
            "No. of footprints": len(df),
            "No. of known individuals": individual_id_col.nunique(),
            "No. of female individuals": n_female,
            "No. of male individuals": n_male,
            "No. of trails": df.get("trail", pd.Series()).nunique(),
            "No. of measurements": len(feature_cols),
        }
        rows.append(row)

    overview_df = pd.DataFrame(rows)
    
    # Add total row at the end
    total_row = {
        "Species": "All Species Total",
        "No. of footprints": overview_df["No. of footprints"].sum(),
        "No. of known individuals": overview_df["No. of known individuals"].sum(),
        "No. of female individuals": overview_df["No. of female individuals"].sum(),
        "No. of male individuals": overview_df["No. of male individuals"].sum(),
        "No. of trails": overview_df["No. of trails"].sum(),
        "No. of measurements": overview_df["No. of measurements"].sum(),
    }
    overview_df = pd.concat([overview_df, pd.DataFrame([total_row])], ignore_index=True)
    
    out_path.parent.mkdir(parents=True, exist_ok=True)
    overview_df.to_parquet(out_path, index=False, compression="gzip")
    return overview_df
