from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
from FIT_python.config import (
    DEFAULT_TARGETS,
    RAW_DIR,
    RESULTS_DATA_DIR,
    SPLITS_DIR,
    GLOBAL_RANDOM_SEED,
)

from FIT_python.data_split_and_summary.data_import_wrapper import DataImporter, DataImportWrapper
from FIT_python.data_split_and_summary.split_utils import (
    create_train_test_split_otter,
    stratified_individual_split, _make_folds
)
from FIT_python.data_split_and_summary.summary_data_wrapper import run_summary

# ==== Config ================================================================
GROUP_COL: str = "individual_id"
STRATIFY_COL: str = "sex"
NUM_FOLDS: int = 3
TEST_SIZE: float = 0.2

# Mapping for column renaming during normalization
COLUMN_RENAME_MAP: Dict[str, str] = {
    "animal": "individual_id",
    "individual": "individual_id",
}

# Species requiring the custom Otter split algorithm
SPECIAL_SPLITS: Dict[str, str] = {"eurasian_otter": "otter"}

# Parameters used by ``create_train_test_split_otter``
OTTER_SPLIT: Dict[str, object] = {
    "inference_origin": "fieldprints_portugal",
    "test_origin": "fieldprints_lower_saxony",
    "sample_origins": {
        "own_data_collection": {"f": 3, "m": 3},
        "vetrecova_et_al": {"f": 2, "m": 2},
    },
}


def normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    """Return a normalized copy of ``df``."""
    df_norm = df.copy()
    df_norm.columns = [
        c.strip().lower().replace(" ", "_").replace(".", "_") for c in df_norm.columns
    ]
    df_norm.rename(columns=COLUMN_RENAME_MAP, inplace=True)

    for col in ["individual_id", "dataorigin", "sex", "trail"]:
        if col in df_norm.columns:
            df_norm[col] = (
                df_norm[col]
                .astype(str)
                .str.strip()
                .str.lower()
                .str.replace(" ", "_")
                .str.replace(".", "_")
            )
    return df_norm


def prepare_all_splits(csv_fp: Path) -> None:
    """
    Erzeugt Splits für genau eine CSV-Datei.

    Parameters
    ----------
    csv_fp : Path
        Pfad zur Eingabe-CSV (z.B. data/raw/Eurasian Otter.csv).
    """
    species = csv_fp.stem  # "Eurasian Otter"
    print(f"→ Preparing splits for: {csv_fp.name!r} (species key = {species!r})")

    # 1) Import & clean
    importer = DataImporter(RAW_DIR, target_cols=DEFAULT_TARGETS)
    dfs = importer.run()
    if DataImportWrapper().clean_all() != 0:
        raise RuntimeError("Data import failed")

    # 2) Pick the right sheet by partial match on species
    # normalize file stem to snake_case
    sp_key = species.strip().lower().replace(" ", "_")
    matches = [k for k in dfs.keys() if sp_key in k.lower()]
    if matches:
        key = matches[0]
    else:
        key = next(iter(dfs))
        print(f"⚠️ Keine exakte Übereinstimmung für '{species}', verwende Key {key!r}")
    df = dfs[key]

    # Normalize columns/labels
    df = normalize_df(df)

    # 3) Choose split strategy
    if SPECIAL_SPLITS.get(sp_key) == "otter":
        train_df, test_df, inf_df = create_train_test_split_otter(
            df,
            inference_origin=OTTER_SPLIT["inference_origin"],
            test_origin=OTTER_SPLIT["test_origin"],
            sample_origins=OTTER_SPLIT["sample_origins"],
            seed=GLOBAL_RANDOM_SEED,
            n_folds=NUM_FOLDS,
        )
    else:
        train_df, test_df, inf_df = stratified_individual_split(
            df,
            test_size=TEST_SIZE,
            random_state=GLOBAL_RANDOM_SEED,
            group_col=GROUP_COL,
            stratify_col=STRATIFY_COL,
            add_folds=True,
            n_folds=NUM_FOLDS,
        )

    # 4) Ensure fold column exists
    if "Fold" not in train_df.columns:
        y_train = train_df[STRATIFY_COL].map({"f": 0, "m": 1})
        folds, _ = _make_folds(
            train_df,
            y_train,
            n_splits=NUM_FOLDS,
            group_col=GROUP_COL,
        )
        train_df = train_df.assign(Fold=folds)

    # 5) Abspeichern
    out_dir = SPLITS_DIR / species.replace(" ", "_").lower()
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, df_out in [("train", train_df), ("test", test_df), ("inference", inf_df)]:
        fp = out_dir / f"{name}.parquet"
        df_out.to_parquet(fp, index=False)
        print(f"   ✔ wrote {fp}")

    # 6) Summary & Plots
    run_summary(
        out_dir,
        RESULTS_DATA_DIR / f"{species.replace(' ','_').lower()}_summary.csv",
        RESULTS_DATA_DIR / f"{species.replace(' ','_').lower()}_fig"
    )
    print(f"✅ Finished preparing splits for {species!r}")
