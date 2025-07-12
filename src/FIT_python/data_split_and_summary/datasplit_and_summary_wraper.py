from pathlib import Path
from typing import List, Optional
import pandas as pd

from FIT_python.data_split_and_summary.data_import_wrapper import DataImporter
from FIT_python.data_split_and_summary.split_utils import (
    create_train_test_split_otter,
    stratified_individual_split,
    _make_folds,
)
from FIT_python.data_split_and_summary.summary_data_wrapper import run_summary

RAW_DIR = Path("data/raw")
SPLITS_DIR = Path("data/splits")
RESULTS_DATA_DIR = Path("results/data")
DEFAULT_TARGETS = [...]  # wie gehabt
NUM_FOLDS = 5
GROUP_COL = "individual_id"


def prepare_all_splits(species_filter: Optional[List[str]] = None) -> None:
    """
    Lädt alle CSVs in RAW_DIR und erstellt für jede Art Splits:

    - Wenn `species_filter` angegeben ist, nur für diese Arten.
    - Für 'Eurasian Otter.csv' wird `create_train_test_split_otter` verwendet.
    - Für alle anderen Arten `stratified_individual_split`.
    """

    for csv_fp in RAW_DIR.glob("*.csv"):
        species = csv_fp.stem  # z.B. "Eurasian Otter"
        # Filter?
        if species_filter and species not in species_filter:
            continue

        # 1) Import
        importer = DataImporter(RAW_DIR, target_cols=DEFAULT_TARGETS)
        dfs = importer.run()
        # Schlüssel finden, der zur aktuellen Datei passt
        # (DataImporter kann mehrere Tabs/Sheets liefern)
        # match against cleaned keys where spaces are converted to underscores
        target_key = species.lower().replace(" ", "_")
        key = next(k for k in dfs if target_key in k.lower())
        df = dfs[key]

        # 2) Splits erstellen
        if species.lower() == "eurasian otter":
            train_df, test_df, inf_df = create_train_test_split_otter(df)
        else:
            # generischer Stratified split nach individual_id (oder GROUP_COL)
            train_df, test_df, inf_df = stratified_individual_split(
                df, group_col=GROUP_COL, n_splits=NUM_FOLDS, random_state=0
            )

        # 3) Folds ins train_df schreiben, falls nicht schon geschehen
        if "Fold" not in train_df.columns:
            y_train = (
                train_df["sex"].map({"f": 0, "m": 1})
                if species.lower() == "eurasian otter"
                else train_df[GROUP_COL]
            )
            folds, _ = _make_folds(
                train_df, y_train, n_splits=NUM_FOLDS, group_col=GROUP_COL
            )
            train_df = train_df.assign(Fold=folds)

        # 4) Abspeichern
        out_dir = SPLITS_DIR / species.replace(" ", "_").lower()
        out_dir.mkdir(parents=True, exist_ok=True)
        train_df.to_parquet(out_dir / "train.parquet", index=False)
        test_df.to_parquet(out_dir / "test.parquet", index=False)
        inf_df.to_parquet(out_dir / "inference.parquet", index=False)

        # 5) Optional: Zusammenfassung & Plots
        run_summary(
            out_dir,
            RESULTS_DATA_DIR / f"{species.replace(' ','_').lower()}_summary.csv",
            RESULTS_DATA_DIR / f"{species.replace(' ','_').lower()}_fig",
        )
