from pathlib import Path
from typing import List, Optional
import pandas as pd
from tqdm.auto import tqdm

from FIT_python.data_split_and_summary.data_import_wrapper import DataImporter
from FIT_python.data_split_and_summary.split_utils import (
    create_train_test_split_otter,
    stratified_individual_split,
    _make_folds,
)
from FIT_python.data_split_and_summary.summary_data_wrapper import run_summary

from FIT_python.config import (
    RAW_DIR,
    SPLITS_DIR,
    DEFAULT_TARGETS,
    GROUP_COL,
    NUM_FOLDS,
    GLOBAL_RANDOM_SEED,
)


class SplitWrapper:
    """High-level interface to create train/test/inference splits and assign folds."""

    def __init__(self, input_dir: Optional[Path] = None, output_dir: Optional[Path] = None) -> None:
        self.input_dir = input_dir or RAW_DIR
        self.output_dir = output_dir or SPLITS_DIR

    def print_summary(self, df: pd.DataFrame, name: str) -> None:
        n_rows = len(df)
        n_individuals = df["individual_id"].nunique() if "individual_id" in df.columns else 0
        n_trails = df["trail"].nunique() if "trail" in df.columns else 0
        sex_counts = (
            df["sex"].fillna("Unknown").astype(str).str.strip().value_counts()
        )
        print(f"\n📊 {name} – {n_rows} Zeilen | {n_individuals} Individuen | {n_trails} Trails")
        print(sex_counts.to_string())

    def split_all(self, reuse_splits: bool = True) -> int:
        importer = DataImporter(raw_dir=self.input_dir, target_cols=DEFAULT_TARGETS)
        dfs = importer.run()
        if not dfs:
            print(f"❌ Keine Rohdaten gefunden in {self.input_dir}")
            return 1

        for name, df in tqdm(dfs.items(), desc="Splitting datasets"):
            dataset = name.lower().replace("_cleaned", "")
            out_dir = self.output_dir / dataset

            if reuse_splits and all((out_dir / f"{s}.parquet").exists() for s in ["train", "test", "inference"]):
                print(f"↪️  Verwende bestehende Splits für {dataset}")
                continue
            species_col = df.get("Species") or df.get("species")
            is_otter = (
                species_col.astype(str)
                .str.strip()
                .str.lower()
                .eq("lutra_lutra")
                .any()
                if species_col is not None
                else False
            )

            if is_otter:
                print(f"🦦 Verwende Otter-Splits für {dataset}")
                train_df, test_df, inf_df = create_train_test_split_otter(df)
            else:
                train_df, test_df, inf_df = stratified_individual_split(
                    df, group_col="individual_id", stratify_col="sex"
                )

            if "Fold" in train_df.columns and train_df["Fold"].notna().all():
                print(
                    f"✔️ Fold-Spalte in {dataset} schon gesetzt – keine erneute Zuweisung."
                )
                fold_method = "predefined"
            else:
                y_ser = train_df["sex"].map({"f": 0, "m": 1})
                fold_ids, fold_method = _make_folds(
                    train_df, y_ser, n_splits=NUM_FOLDS, group_col=GROUP_COL
                )
                train_df = train_df.assign(Fold=fold_ids)

            print(f"✅ Fold-Methode verwendet: {fold_method}")
            print(
                train_df["Fold"].value_counts().sort_index().rename("count").to_string()
            )

            out_dir = self.output_dir / dataset
            out_dir.mkdir(parents=True, exist_ok=True)

            for split_name, split_df in zip(
                ["train", "test", "inference"], [train_df, test_df, inf_df]
            ):
                split_df.to_parquet(
                    out_dir / f"{split_name}.parquet", index=False, compression="gzip"
                )
                self.print_summary(split_df, f"{dataset} – {split_name}")

            print(f"\n✅ {dataset} gespeichert (parquet)")

        return 0




def prepare_all_splits(species_filter: Optional[List[str]] = None) -> None:
    """
    Lädt alle CSVs in RAW_DIR und erstellt für jede Art Splits:

    - Wenn `species_filter` angegeben ist, nur für diese Arten.
    - Für 'Eurasian Otter.csv' wird `create_train_test_split_otter` verwendet.
    - Für alle anderen Arten `stratified_individual_split`.
    """

    normalised_filter = None
    if species_filter:
        normalised_filter = {
            s.replace(" ", "_").lower() for s in species_filter
        }

    for csv_fp in RAW_DIR.glob("*.csv"):
        species = csv_fp.stem  # z.B. "Eurasian Otter"
        species_key = species.replace(" ", "_").lower()
        # Filter?
        if normalised_filter and species_key not in normalised_filter:
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
                df, group_col=GROUP_COL, n_folds=NUM_FOLDS, random_state=GLOBAL_RANDOM_SEED
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
        run_summary(out_dir, species.replace(" ", "_" ).lower())
