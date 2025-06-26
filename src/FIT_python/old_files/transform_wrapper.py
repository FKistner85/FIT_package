# src/FIT_python/transform_wrapper.py

from pathlib import Path
import pandas as pd
import numpy as np
import logging
import FIT_python.config as config
from FIT_python.config import DEFAULT_TARGETS, OTTER_META_COLS
from ..transform_utils import convert_numeric

class TransformWrapper:
    """Numeric transformation of train/test splits into NumPy arrays."""

    def __init__(self, logger: logging.Logger | None = None):
        self.logger = logger or logging.getLogger(__name__)

    def transform_dataset(self, dataset_name: str, train_path: Path, test_path: Path) -> None:
        """Transform a single dataset given train and test parquet paths."""

        dataset = dataset_name.replace(" ", "_")
        out_dir = config.NUMERIC_DIR / dataset
        out_dir.mkdir(parents=True, exist_ok=True)

        for split, pq_path in ("train", train_path), ("test", test_path):
            df = pd.read_parquet(pq_path)

            meta_cols = OTTER_META_COLS if "otter" in dataset.lower() else ["id"]
            feature_cols = [c for c in df.columns if c not in meta_cols + DEFAULT_TARGETS]

            df = convert_numeric(df, feature_cols)
            X = df[feature_cols].to_numpy(dtype=float)

            sex = (
                df["sex"].fillna("unknown")
                .astype(str)
                .str.strip()
                .str.lower()
                .map({
                    "f": "female",
                    "female": "female",
                    "m": "male",
                    "male": "male",
                })
                .fillna("unknown")
            )
            label_map = {"female": 0, "male": 1, "unknown": 2}
            y = sex.map(label_map).to_numpy(dtype=int)

            np.save(out_dir / f"X_{split}.npy", X)
            np.save(out_dir / f"y_{split}.npy", y)
            self.logger.info("Saved %s/%s", dataset, split)

    def transform_all(self) -> None:
        """Transform all datasets found in :data:`SPLITS_DIR`."""

        config.NUMERIC_DIR.mkdir(parents=True, exist_ok=True)

        for ds_folder in config.SPLITS_DIR.iterdir():
            if not ds_folder.is_dir():
                continue

            name = ds_folder.name.replace(" ", "_")
            train_pq = ds_folder / "train.parquet"
            test_pq = ds_folder / "test.parquet"

            missing = None
            for path in (train_pq, test_pq):
                if not path.exists():
                    missing = path
                    break
            if missing:
                msg = f"Required file not found: {missing}"
                if config.DEBUG_MODE:
                    raise FileNotFoundError(msg)
                else:
                    self.logger.warning("Skipping dataset: %s", msg)
                    continue

            self.transform_dataset(name, train_pq, test_pq)
