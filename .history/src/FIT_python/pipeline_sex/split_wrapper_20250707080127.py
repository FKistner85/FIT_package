from pathlib import Path
import pandas as pd
from typing import Optional

from FIT_python.pipeline_sex.data_import_wrapper import DataImporter
from FIT_python.pipeline_sex.split_utils import (
    create_train_test_split_otter,
    train_test_group_split,
    _make_folds,
)
from FIT_python.config import (
    RAW_DIR,
    SPLITS_DIR,
    DEFAULT_TARGETS,
    GLOBAL_RANDOM_SEED,
    NUM_FOLDS,
    GROUP_COL,
)


class SplitWrapper:
    """Wrapper, der Rohdaten bereinigt, splittet und Folds robust erzeugt."""

    def __init__(
        self,
        input_dir: Optional[Path] = None,
        output_dir: Optional[Path] = None,
    ):
        self.input_dir = input_dir or RAW_DIR
        self.output_dir = output_dir or SPLITS_DIR

    def print_summary(self, df: pd.DataFrame, name: str):
        n_rows = len(df)
        n_individuals = (
            df["individual_id"].nunique()
            if "individual_id" in df.columns
            else 0
        )
        n_trails = df["trail"].nunique() if "trail" in df.columns else 0
        sex_counts = (
            df["sex"]
            .fillna("Unknown")
            .astype(str)
            .str.strip()
            .value_counts()
        )
        print(
            f"\n📊 {name} – {n_rows} Zeilen | "
            f"{n_individuals} Individuen | {n_trails} Trails"
        )
        print(sex_counts.to_string())

    def split_all(self, as_csv: bool = True) -> int:
        # 1) Lade und bereinige mit DataImporter
        importer = DataImporter(
            raw_dir=self.input_dir, target_cols=DEFAULT_TARGETS
        )
        dfs = importer.run()
        if not dfs:
            print(f"❌ Keine Rohdaten gefunden in {self.input_dir}")
            return 1

        # 2) Für jeden bereinigten DataFrame: Split + Fold
        for name, df in dfs.items():
            dataset = name.lower().replace("_cleaned", "")

            # 2a) OTTER-Spezialfall: fixed Split
            species_col = df.get("Species") or df.get("species")
            is_otter = (
                species_col.astype(str)
                .str.strip()
                .str.lower()
                .eq("lutra_lutra")
                .any()
            )

            if is_otter:
                print(f"🦦 Verwende Otter-Splits für {dataset}")
                train_df, test_df, inference_df = create_train_test_split_otter(df)
            else:
                train_df, test_df, inference_df = stratified_group_split(df, group_col="individual_id", stratify_col="sex")

            # 3) Fold-Generierung für train_df – nur wenn nicht vorhanden
            if "Fold" in train_df.columns and train_df["Fold"].notna().all():
                print(f"✔️ Fold-Spalte in {dataset} schon gesetzt – keine erneute Zuweisung.")
                fold_method = "predefined"
            else:
                y_ser = train_df["sex"].map({"f": 0, "m": 1})
                fold_ids, fold_method = _make_folds(
                    train_df, y_ser, n_splits=NUM_FOLDS, group_col=GROUP_COL
                )
                train_df = train_df.assign(Fold=fold_ids)

            print(f"✅ Fold-Methode verwendet: {fold_method}")
            print(train_df["Fold"].value_counts().sort_index().rename("count").to_string())

            # 4) Speichern & Summary
            out_dir = self.output_dir / dataset
            out_dir.mkdir(parents=True, exist_ok=True)

            for split_name, split_df in zip(
                ["train", "test", "inference"],
                [train_df, test_df, inference_df],
            ):
                split_df.to_parquet(out_dir / f"{split_name}.parquet", index=False)
                if as_csv:
                    split_df.to_csv(out_dir / f"{split_name}.csv", index=False)
                self.print_summary(split_df, f"{dataset} – {split_name}")

            print(f"\n✅ {dataset} gespeichert (csv & parquet)")

        return 0
