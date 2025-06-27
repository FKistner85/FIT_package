from __future__ import annotations

"""Wrapper that cleans raw data, performs splits and assigns folds."""

from pathlib import Path
from typing import Optional
import pandas as pd

from FIT_python.data_import_wrapper import DataImporter
from FIT_python.old_files.split_utils import (
    create_train_test_split_otter,
    train_test_group_split,
)
import FIT_python.config as config

from .split_utils import _make_folds


class SplitWrapper:
    """Wrapper that cleans raw data, creates splits and robust folds."""

    def __init__(self, input_dir: Optional[Path] = None, output_dir: Optional[Path] = None):
        self.input_dir = input_dir or config.RAW_DIR
        self.output_dir = output_dir or config.PROCESSED_SPLITS_DIR

    def print_summary(self, df: pd.DataFrame, name: str) -> None:
        n_rows = len(df)
        n_individuals = df["individual_id"].nunique() if "individual_id" in df.columns else 0
        n_trails = df["trail"].nunique() if "trail" in df.columns else 0
        sex_counts = (
            df["sex"].fillna("Unknown").astype(str).str.strip().value_counts()
        )
        print(
            f"\n📊 {name} – {n_rows} Zeilen | "
            f"{n_individuals} Individuen | {n_trails} Trails"
        )
        print(sex_counts.to_string())

    def split_all(self, as_csv: bool = True) -> int:
        importer = DataImporter(raw_dir=self.input_dir, target_cols=config.DEFAULT_TARGETS)
        dfs = importer.run()
        if not dfs:
            print(f"❌ Keine Rohdaten gefunden in {self.input_dir}")
            return 1

        for name, df in dfs.items():
            dataset = name.lower().replace("_cleaned", "")

            species_col = df.get("Species") or df.get("species")
            if species_col is not None:
                is_otter = (
                    species_col.astype(str)
                    .str.strip()
                    .str.lower()
                    .eq("lutra_lutra")
                    .any()
                )
            else:
                is_otter = False

            if is_otter:
                train_df, test_df, inference_df = create_train_test_split_otter(df)

                def valid_split(dd):
                    vals = set(dd["sex"].dropna().str.lower())
                    return {"f", "m"}.issubset(vals)

                if not (
                    valid_split(train_df)
                    and valid_split(test_df)
                    and valid_split(inference_df)
                ):
                    print(f"⚠️ OTTER-Splits für {dataset} ungültig, fallback auf Gruppen-Split")
                    train_df, test_df = train_test_group_split(df)
                    inference_df = df.loc[
                        ~df["individual_id"].isin(train_df["individual_id"])
                        & ~df["individual_id"].isin(test_df["individual_id"])
                    ]
            else:
                train_df, test_df = train_test_group_split(df)
                inference_df = df.loc[
                    ~df["individual_id"].isin(train_df["individual_id"])
                    & ~df["individual_id"].isin(test_df["individual_id"])
                ]

            y_ser = train_df["sex"].map({"f": 0, "m": 1})
            fold_ids, fold_method = _make_folds(
                train_df, y_ser, n_splits=config.NUM_FOLDS, group_col=config.GROUP_COL
            )
            train_df = train_df.assign(Fold=fold_ids)
            print(f"✔️ Fold-Split für {dataset} mit Methode '{fold_method}'")

            self.output_dir.mkdir(parents=True, exist_ok=True)

            for split_name, split_df in zip(
                ["train", "test", "inference"],
                [train_df, test_df, inference_df],
            ):
                out_path = self.output_dir / f"{dataset}_{split_name}.parquet"
                split_df.to_parquet(out_path, index=False)
                if as_csv:
                    csv_path = self.output_dir / f"{dataset}_{split_name}.csv"
                    split_df.to_csv(csv_path, index=False)
                self.print_summary(split_df, f"{dataset} – {split_name}")

            print(f"\n✅ {dataset} gespeichert (csv & parquet)")

        return 0
