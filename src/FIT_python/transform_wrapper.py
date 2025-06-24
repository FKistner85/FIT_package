# src/FIT_python/transform_wrapper.py

from pathlib import Path
import pandas as pd
import numpy as np
import joblib
from FIT_python.config import (
    SPLITS_DIR,
    PROCESSED_DIR,
    NUMERIC_DIR,
    DEFAULT_TARGETS,
    OTTER_META_COLS
)
from .transform_utils import convert_numeric, one_hot_encode_targets, save_target_mapping

class TransformWrapper:
    """
    Applies numeric conversion and one-hot encoding to processed splits.
    Saves numpy arrays and mappings under NUMERIC_DIR/<Dataset>/.
    """
    def __init__(self):
        pass

    def transform_all(self):
        NUMERIC_DIR.mkdir(parents=True, exist_ok=True)
        for ds_folder in PROCESSED_DIR.iterdir():
            if not ds_folder.is_dir():
                continue
            dataset = ds_folder.name
            out_dir = NUMERIC_DIR / dataset
            out_dir.mkdir(parents=True, exist_ok=True)

            for split in ['train', 'test']:
                parquet_path = ds_folder / f"{split}.parquet"
                if not parquet_path.exists():
                    continue
                df = pd.read_parquet(parquet_path)
                # determine feature columns
                if 'otter' in dataset.lower():
                    meta = OTTER_META_COLS
                else:
                    meta = ['id']
                features = [c for c in df.columns if c not in meta + DEFAULT_TARGETS]
                # convert numeric
                df_num = convert_numeric(df.copy(), features)
                X = df_num[features].to_numpy()
                # encode targets
                y, mapping = one_hot_encode_targets(df_num, DEFAULT_TARGETS)
                # save arrays
                np.save(out_dir / f"X_{split}.npy", X)
                np.save(out_dir / f"y_{split}.npy", y)
                # save mapping once
                save_target_mapping(mapping, out_dir / 'target_mapping.json')
                print(f"Saved numeric arrays for {dataset}/{split} in {out_dir}")

        print(f"All transformations saved under {NUMERIC_DIR}")
