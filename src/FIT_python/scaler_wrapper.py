# src/FIT_python/scaler_wrapper.py

from pathlib import Path
import pandas as pd
import numpy as np
import joblib

import FIT_python.config as config
from FIT_python.config import (
    DEFAULT_TARGETS,
    OTTER_META_COLS,
    DEFAULT_SCALER,
    SCALER_PARAMS,
    normalize_dataset_name,
)
from FIT_python.scaler_utils import get_standard_scaler, get_robust_scaler
from FIT_python.transform_utils import convert_numeric

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
        config.SCALED_DIR.mkdir(parents=True, exist_ok=True)

        datasets = {
            p.stem.rsplit("_", 1)[0]
            for p in config.PROCESSED_SPLITS_DIR.glob("*_train.parquet")
        }

        for dataset in sorted(datasets):
            print(f"\nScaling dataset: {dataset}")

            out_folder = config.SCALED_DIR
            out_folder.mkdir(parents=True, exist_ok=True)

            # instantiate scaler once per dataset
            scaler = self._make_scaler()

            for split in ("train", "test"):
                src = config.PROCESSED_SPLITS_DIR / f"{dataset}_{split}.parquet"
                if not src.exists():
                    msg = f"Required file not found: {src}"
                    if config.DEBUG_MODE:
                        raise FileNotFoundError(msg)
                    else:
                        print("Skipping:", msg)
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

                # convert numeric columns before scaling
                df = convert_numeric(df, feature_cols)

                # apply scaling if requested
                if scaler:
                    if split == "train":
                        scaled = scaler.fit_transform(df[feature_cols])
                        # save fitted scaler
                        self.scalers[dataset] = scaler
                        joblib.dump(
                            scaler,
                            config.SCALED_DIR / f"{dataset}_scaler.pkl"
                        )
                    else:
                        scaled = scaler.transform(df[feature_cols])
                    # assign back scaled values
                    df.loc[:, feature_cols] = scaled

                # write out the scaled DataFrame
                dest = out_folder / f"{dataset}_{split}.parquet"
                df.to_parquet(dest, index=False)
                print(f"  Saved scaled {split}: {dest} ({len(df)} rows)")

            # also persist numeric numpy arrays
            np_dir = config.NUMERIC_DIR / dataset
            np_dir.mkdir(parents=True, exist_ok=True)
            np.save(np_dir / "X_train.npy", pd.read_parquet(out_folder / f"{dataset}_train.parquet")[feature_cols].to_numpy())
            np.save(np_dir / "y_train.npy", pd.read_parquet(out_folder / f"{dataset}_train.parquet")[["sex"]].to_numpy())
            np.save(np_dir / "X_test.npy", pd.read_parquet(out_folder / f"{dataset}_test.parquet")[feature_cols].to_numpy())
            np.save(np_dir / "y_test.npy", pd.read_parquet(out_folder / f"{dataset}_test.parquet")[["sex"]].to_numpy())

        print(
            f"\nAll datasets processed. Scaled files are under '{config.SCALED_DIR}'"
        )
