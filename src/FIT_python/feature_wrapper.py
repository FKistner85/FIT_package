# src/FIT_python/feature_wrapper.py

from pathlib import Path
from typing import List, Optional
import numpy as np
import pandas as pd
import sys

import FIT_python.config as config
from FIT_python.config import DEFAULT_TARGETS, OTTER_META_COLS
from FIT_python.feature_utils import run_feature_selection_methods

class FeatureSelector:
    """Run multiple feature selection strategies on scaled datasets."""

    def __init__(
        self,
        scaled_dir: Path | None = None,
        out_dir: Path | None = None,
        methods: Optional[List[str]] = None,
    ) -> None:
        self.scaled_dir = scaled_dir or config.SCALED_DIR
        self.out_dir = out_dir or config.FEATURE_SELECTED_DIR
        self.methods = methods or config.FS_DEFAULT_METHODS

    def select_all(self) -> int:
        if not self.scaled_dir.exists():
            print(f"[ERROR] {self.scaled_dir} not found", file=sys.stderr)
            if config.DEBUG_MODE:
                raise FileNotFoundError(self.scaled_dir)
            return 1

        self.out_dir.mkdir(parents=True, exist_ok=True)
        datasets = {p.stem.rsplit("_", 1)[0] for p in self.scaled_dir.glob("*_train.parquet")}

        for dataset in datasets:
            train_p = self.scaled_dir / f"{dataset}_train.parquet"
            test_p = self.scaled_dir / f"{dataset}_test.parquet"
            df_train = pd.read_parquet(train_p)
            df_test = pd.read_parquet(test_p)

            meta_cols = OTTER_META_COLS if "otter" in dataset.lower() else ["id"]
            feature_cols = [c for c in df_train.columns if c not in meta_cols + DEFAULT_TARGETS]

            results = run_feature_selection_methods(
                df_train[feature_cols],
                df_train["sex"],
                methods=self.methods,
                target_feature_counts=config.FS_TARGET_FEATURE_COUNTS,
                verbose=False,
            )

            for method_name, feats in results.items():
                out_meth = self.out_dir / method_name
                out_meth.mkdir(parents=True, exist_ok=True)
                for split_name, df in [("train", df_train), ("test", df_test)]:
                    out_path = out_meth / f"{dataset}_{split_name}.parquet"
                    df_sel = df[meta_cols + DEFAULT_TARGETS + feats]
                    df_sel.to_parquet(out_path, index=False)
                print(f"[REPORT] {dataset} {method_name}: {len(feats)} features")

            # also write backward compatible file with variance top features
            variances = df_train[feature_cols].var().sort_values(ascending=False)
            var_feats = variances.head(getattr(config, "N_FEATURES", 2)).index.tolist()
            for split_name, df in [("train", df_train), ("test", df_test)]:
                out_path = self.out_dir / f"{dataset}_{split_name}.parquet"
                df_sel = df[meta_cols + DEFAULT_TARGETS + var_feats]
                df_sel.to_parquet(out_path, index=False)

        print("[SUCCESS] feature selection completed.")
        return 0
