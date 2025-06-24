# src/FIT_python/feature_wrapper.py

from pathlib import Path
import numpy as np
import pandas as pd

from FIT_python.config import (
    NUMERIC_DIR,
    PROCESSED_DIR,
    DEFAULT_TARGETS,
    OTTER_META_COLS
)
from FIT_python.feature_utils import run_feature_selection_methods

class FeatureSelector:
    """
    Wrapper to apply feature selection methods to numeric datasets,
    for each target column in DEFAULT_TARGETS.
    """

    def __init__(self):
        pass

    def select_all(self):
        for ds_folder in NUMERIC_DIR.iterdir():
            if not ds_folder.is_dir():
                continue
            dataset = ds_folder.name
            print(f"\n=== Feature selection for {dataset} ===")

            out_base = PROCESSED_DIR / dataset / "feature_selection"
            out_base.mkdir(parents=True, exist_ok=True)

            # load numeric X once
            X = np.load(ds_folder / "X_train.npy", allow_pickle=True)
            df_raw = pd.read_parquet(PROCESSED_DIR / dataset / "train.parquet")
            # determine feature names
            if "otter" in dataset.lower():
                meta = OTTER_META_COLS
            else:
                meta = ["id"]
            feature_cols = [c for c in df_raw.columns if c not in meta + DEFAULT_TARGETS]
            X_df = pd.DataFrame(X, columns=feature_cols)

            # jetzt für jedes Target
            for target in DEFAULT_TARGETS:
                print(f"\n-- Target: {target} --")
                # lade das originale label-series
                y_series = df_raw[target]

                # run selection
                results = run_feature_selection_methods(X_df, y_series)

                # speichere pro Methode eine Datei unter feature_selection/<target>/
                out_dir = out_base / target
                out_dir.mkdir(parents=True, exist_ok=True)
                for method, feats in results.items():
                    path = out_dir / f"{method}.txt"
                    pd.Series(feats).to_csv(path, index=False, header=False)
                    print(f" Saved {target}/{method}.txt ({len(feats)} features)")
