# src/FIT_python/transform_wrapper.py

from pathlib import Path
import pandas as pd
import numpy as np
import logging
from FIT_python.config import (
    PROCESSED_DIR,
    NUMERIC_DIR,
    DEFAULT_TARGETS,
    OTTER_META_COLS,
    normalize_dataset_name,
    DEBUG_MODE,
)
from .transform_utils import convert_numeric, one_hot_encode_targets, save_target_mapping

class TransformWrapper:
    """
    Applies numeric conversion and one-hot encoding to processed splits.
    Saves numpy arrays and mappings under NUMERIC_DIR/<Dataset>/.
    """
    def __init__(self, logger: logging.Logger | None = None):
        self.logger = logger or logging.getLogger(__name__)

    def transform_all(self):
        NUMERIC_DIR.mkdir(parents=True, exist_ok=True)
        successes = []
        failures = []
        for ds_folder in PROCESSED_DIR.iterdir():
            if not ds_folder.is_dir():
                continue
            dataset = normalize_dataset_name(ds_folder.name)
            out_dir = NUMERIC_DIR / dataset
            out_dir.mkdir(parents=True, exist_ok=True)

            try:
                for split in ["train", "test"]:
                    parquet_path = ds_folder / f"{split}.parquet"
                    if not parquet_path.exists():
                        msg = f"Required file not found: {parquet_path}"
                        if DEBUG_MODE:
                            raise FileNotFoundError(msg)
                        else:
                            self.logger.warning("Skipping: %s", msg)
                            raise RuntimeError("skip")
                    df = pd.read_parquet(parquet_path)
                    if "otter" in dataset.lower():
                        meta = OTTER_META_COLS
                    else:
                        meta = ["id"]
                    features = [c for c in df.columns if c not in meta + DEFAULT_TARGETS]
                    df_num = convert_numeric(df.copy(), features)
                    X = df_num[features].to_numpy()
                    y, mapping = one_hot_encode_targets(df_num, DEFAULT_TARGETS)
                    np.save(out_dir / f"X_{split}.npy", X)
                    np.save(out_dir / f"y_{split}.npy", y)
                    save_target_mapping(mapping, out_dir / "target_mapping.json")
                    self.logger.info("Saved %s/%s", dataset, split)
                successes.append(dataset)
            except RuntimeError:
                # skipped due to missing file in non-debug mode
                continue
            except Exception as e:
                failures.append(ds_folder.name)
                self.logger.warning("Failed to transform %s: %s", ds_folder.name, e)

        self.logger.info("Transformation complete. Success: %s", successes)
        if failures:
            self.logger.warning("Datasets failed: %s", failures)
