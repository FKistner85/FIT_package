# src/FIT_python/scaler_wrapper.py

from pathlib import Path
import pandas as pd
import joblib

from FIT_python.config import (
    SPLITS_DIR,
    PROCESSED_DIR,
    DEFAULT_TARGETS,
    OTTER_META_COLS,
    DEFAULT_SCALER,
    SCALER_PARAMS
)
from FIT_python.scaler_utils import get_standard_scaler, get_robust_scaler

class ScalerWrapper:
    """
    Applies scaling (none, standard, robust) to train/test Parquet splits.
    Reads from SPLITS_DIR/<Dataset>/{train,test}.parquet and writes
    scaled versions to PROCESSED_DIR/<Dataset>/{train,test}.parquet.
    """

    def __init__(self, scaler_type: str = None):
        self.scaler_type = (scaler_type or DEFAULT_SCALER).lower()
        self.scalers = {}  # to store fitted scaler per dataset

    def _make_scaler(self):
        if self.scaler_type == "standard":
            return get_standard_scaler(SCALER_PARAMS.get("standard", {}))
        elif self.scaler_type == "robust":
            return get_robust_scaler(SCALER_PARAMS.get("robust", {}))
        # raw / none
        return None

    def scale_all(self):
        # Ensure processed base exists
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

        for ds_folder in SPLITS_DIR.iterdir():
            if not ds_folder.is_dir():
                continue

            dataset = ds_folder.name  # e.g. "Giant_Panda"
            print(f"\nScaling dataset: {dataset}")

            # prepare output subfolder
            out_folder = PROCESSED_DIR / dataset
            out_folder.mkdir(parents=True, exist_ok=True)

            # instantiate scaler once per dataset
            scaler = self._make_scaler()

            for split in ("train", "test"):
                src = ds_folder / f"{split}.parquet"
                if not src.exists():
                    print(f"  Skipping {dataset}/{split}: file not found.")
                    continue

                # load the split
                df = pd.read_parquet(src)

                # determine which columns to scale
                if "otter" in dataset.lower():
                    meta_cols = OTTER_META_COLS
                else:
                    meta_cols = ["id"]
                target_cols = DEFAULT_TARGETS

                feature_cols = [
                    c for c in df.columns
                    if c not in meta_cols + target_cols
                ]

                # apply scaling if requested
                if scaler:
                    if split == "train":
                        scaled = scaler.fit_transform(df[feature_cols])
                        # save fitted scaler
                        self.scalers[dataset] = scaler
                        joblib.dump(
                            scaler,
                            out_folder / f"{dataset}_scaler.pkl"
                        )
                    else:
                        scaled = scaler.transform(df[feature_cols])
                    # assign back scaled values
                    df.loc[:, feature_cols] = scaled

                # write out the scaled DataFrame
                dest = out_folder / f"{split}.parquet"
                df.to_parquet(dest, index=False)
                print(f"  Saved scaled {split}: {dest} ({len(df)} rows)")

        print(f"\nAll datasets processed. Scaled files are under '{PROCESSED_DIR}'")
