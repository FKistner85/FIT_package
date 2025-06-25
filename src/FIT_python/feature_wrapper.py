# src/FIT_python/feature_wrapper.py

from pathlib import Path
import numpy as np
import pandas as pd
import sys

import FIT_python.config as config
from FIT_python.config import DEFAULT_TARGETS, OTTER_META_COLS
from FIT_python.feature_utils import run_feature_selection_methods

class FeatureSelector:
    """Wrapper to apply feature selection methods to numeric datasets
    using only the ``sex`` target column.
    """

    def __init__(self):
        pass

    def select_all(self) -> int:
        if not config.SCALED_DIR.exists():
            print(f"[ERROR] {config.SCALED_DIR} not found", file=sys.stderr)
            if config.DEBUG_MODE:
                raise FileNotFoundError(config.SCALED_DIR)
            return 1

        config.FEATURE_SELECTED_DIR.mkdir(parents=True, exist_ok=True)
        datasets = {p.stem.rsplit("_", 1)[0] for p in config.SCALED_DIR.glob("*_train.parquet")}

        for dataset in datasets:
            train_p = config.SCALED_DIR / f"{dataset}_train.parquet"
            test_p = config.SCALED_DIR / f"{dataset}_test.parquet"
            df_train = pd.read_parquet(train_p)
            df_test = pd.read_parquet(test_p)

            meta_cols = OTTER_META_COLS if "otter" in dataset.lower() else ["id"]
            feature_cols = [c for c in df_train.columns if c not in meta_cols + DEFAULT_TARGETS]

            variances = df_train[feature_cols].var().sort_values(ascending=False)
            n_feats = getattr(config, "N_FEATURES", 2)
            selected = variances.head(n_feats).index.tolist()

            for split_name, df in [("train", df_train), ("test", df_test)]:
                out_path = config.FEATURE_SELECTED_DIR / f"{dataset}_{split_name}.parquet"
                df_sel = df[meta_cols + DEFAULT_TARGETS + selected]
                df_sel.to_parquet(out_path, index=False)
            print(f"[REPORT] {dataset} selected features: {selected}")

        print("[SUCCESS] feature selection completed.")
        return 0
