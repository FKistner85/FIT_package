from pathlib import Path
from typing import List, Optional
import pandas as pd
from FIT_python.config import DEFAULT_TARGETS, RAW_DIR, SPLITS_DIR, RESULTS_DATA_DIR, NUM_FOLDS, GROUP_COL

from FIT_python.data_split_and_summary.data_import_wrapper import DataImporter, DataImportWrapper
from FIT_python.data_split_and_summary.split_utils import (
    create_train_test_split_otter,
    stratified_individual_split
)
from FIT_python.data_split_and_summary.summary_data_wrapper import run_summary


def prepare_all_splits(csv_fp: Path) -> None:
    """
    Erzeugt Splits für genau eine CSV-Datei.

    Parameters
    ----------
    csv_fp : Path
        Pfad zur Eingabe-CSV (z.B. data/raw/Eurasian Otter.csv).
    """
    species = csv_fp.stem  # Dateiname ohne .csv, z.B. "Eurasian Otter"
    print(f"→ Preparing splits for: {csv_fp.name!r} (species key = {species!r})")

    # 1) Import
    importer = DataImporter(RAW_DIR, target_cols=DEFAULT_TARGETS)
    dfs = importer.run()
    
    success = DataImportWrapper().clean_all()
    if success != 0:
        raise RuntimeError("Data import failed")
    # Key finden: Teilstring-Match auf species, sonst erster Key
    matches = [k for k in dfs.keys() if species.lower() in k.lower()]
    if matches:
        key = matches[0]
    else:
        key = next(iter(dfs))
        print(f"⚠️ Keine exakte Übereinstimmung für '{species}', verwende Key {key!r}")
    df = dfs[key]

    # 2) Splits erzeugen
    if species.lower() == "eurasian otter":
        train_df, test_df, inf_df = create_train_test_split_otter(df)
    else:
        train_df, test_df, inf_df = stratified_individual_split(
            df, id_col=GROUP_COL, n_splits=NUM_FOLDS, random_state=0
        )

    # 3) Fold-Spalte ergänzen, falls fehlt
    if "Fold" not in train_df.columns:
        if species.lower() == "eurasian_otter":
            y_train = train_df["sex"].map({"f": 0, "m": 1})
        else:
            y_train = train_df[GROUP_COL]
        folds, _ = _make_folds(
            train_df, y_train, n_splits=NUM_FOLDS, group_col=GROUP_COL
        )
        train_df = train_df.assign(Fold=folds)

    # 4) Abspeichern
    out_dir = SPLITS_DIR / species.replace(" ", "_").lower()
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, df_out in [("train", train_df), ("test", test_df), ("inference", inf_df)]:
        fp = out_dir / f"{name}.parquet"
        df_out.to_parquet(fp, index=False)
        print(f"   ✔ wrote {fp}")

    # 5) Summary & Plots
    run_summary(
        out_dir,
        RESULTS_DATA_DIR / f"{species.replace(' ','_').lower()}_summary.csv",
        RESULTS_DATA_DIR / f"{species.replace(' ','_').lower()}_fig"
    )
    print(f"✅ Finished preparing splits for {species!r}")
