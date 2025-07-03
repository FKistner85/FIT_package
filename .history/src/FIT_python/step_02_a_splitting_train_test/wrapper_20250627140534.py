# src/FIT_python/step_02_a_splitting_train_test/wrapper.py

from pathlib import Path
import pandas as pd
from typing import Optional

from FIT_python.data_import_wrapper import DataImporter
from FIT_python.step_02_a_splitting_train_test.utils import (
    create_train_test_split_otter,
    train_test_group_split,
    group_stratified_kfold,
)
from FIT_python.config import RAW_DIR, SPLITS_DIR, DEFAULT_TARGETS


class SplitWrapper:
    """Wrapper, der Rohdaten bereinigt, split­t und Folds robust erzeugt."""

    def __init__(self,
                 input_dir: Optional[Path] = None,
                 output_dir: Optional[Path] = None):
        self.input_dir = input_dir or RAW_DIR
        self.output_dir = output_dir or SPLITS_DIR

    def print_summary(self, df: pd.DataFrame, name: str):
        n_rows = len(df)
        n_individuals = df["individual_id"].nunique() if "individual_id" in df.columns else 0
        n_trails = df["trail"].nunique() if "trail" in df.columns else 0
        sex_counts = (
            df["sex"]
            .fillna("Unknown")
            .astype(str)
            .str.strip()
            .value_counts()
        )
        print(f"\n📊 {name} – {n_rows} Zeilen | "
              f"{n_individuals} Individuen | {n_trails} Trails")
        print(sex_counts.to_string())

    def split_all(self, as_csv: bool = True) -> int:
        # 1) Lade und bereinige mit DataImporter
        importer = DataImporter(raw_dir=self.input_dir,
                                target_cols=DEFAULT_TARGETS)
        dfs = importer.run()
        if not dfs:
            print(f"❌ Keine Rohdaten gefunden in {self.input_dir}")
            return 1

        # 2) Für jeden bereinigten DataFrame Split und Fold
        for name, df in dfs.items():
            dataset = name.lower().replace("_cleaned", "")

            # Otter-Check per Species-Spalte
            species = df.get("Species") or df.get("species")
            is_otter = (
                species
                .astype(str)
                .str.strip()
                .str.lower()
                .eq("lutra lutra")
                .any()
            )

            if is_otter:
                train_df, test_df, inference_df = create_train_test_split_otter(df)
            else:
                train_df, test_df = train_test_group_split(df)
                inference_df = df.loc[
                    ~df["individual_id"].isin(train_df["individual_id"]) &
                    ~df["individual_id"].isin(test_df["individual_id"])
                ]

            # 3) Robust: Versuche StratifiedGroupKFold, sonst Fold=0
            try:
                train_df = group_stratified_kfold(train_df)
            except ValueError as e:
                print(f"⚠️ Kein Fold-Split möglich für {dataset}: {e}")
                train_df["Fold"] = 0

            # 4) Speichern & Summary
            out_dir = self.output_dir / dataset
            out_dir.mkdir(parents=True, exist_ok=True)

            for split_name, split_df in zip(
                ["train", "test", "inference"],
                [train_df, test_df, inference_df]
            ):
                split_df.to_parquet(out_dir / f"{split_name}.parquet", index=False)
                if as_csv:
                    split_df.to_csv(out_dir / f"{split_name}.csv", index=False)
                self.print_summary(split_df, f"{dataset} – {split_name}")

            print(f"\n✅ {dataset} gespeichert (csv & parquet)")

        return 0
