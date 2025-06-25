# src/FIT_python/dim_reduction_wrapper.py
"""Wrapper to apply dimensionality reduction on multiple datasets."""

from pathlib import Path
from typing import Dict, List
import pandas as pd

import FIT_python.config as config
from FIT_python.config import DEFAULT_TARGETS, OTTER_META_COLS
import sys
from FIT_python.dim_reduction_utils import reduce_pca, reduce_tsne, reduce_umap

METHOD_MAP = {
    "pca": reduce_pca,
    "tsne": reduce_tsne,
    "umap": reduce_umap,
}


def reduce_all(X_dict: Dict[str, pd.DataFrame], methods: List[str], n_components: int) -> None:
    """Apply reduction methods to datasets and save Parquet outputs."""
    for dataset, X in X_dict.items():
        out_dir = PROCESSED_DIR / dataset / "dim_reduced"
        out_dir.mkdir(parents=True, exist_ok=True)
        for method in methods:
            func = METHOD_MAP.get(method.lower())
            if func is None:
                raise ValueError(f"Unknown method '{method}'")
            out_path = out_dir / f"{method}_{n_components}.parquet"
            if out_path.exists():
                print(f"Skipping {out_path}, already exists.")
                continue
            df_red = func(X, n_components)
            df_red.to_parquet(out_path, index=False)
            print(f"Saved {out_path}")


class DimReducer:
    """Simple PCA-based dimensionality reduction wrapper."""

    def __init__(self) -> None:
        self.n_components = getattr(config, "N_COMPONENTS", 2)

    def reduce_all(self) -> int:
        if not config.FEATURE_SELECTED_DIR.exists():
            msg = f"Required directory not found: {config.FEATURE_SELECTED_DIR}"
            print(f"[ERROR] {msg}", file=sys.stderr)
            if config.DEBUG_MODE:
                raise FileNotFoundError(msg)
            return 1
        config.DIM_REDUCED_DIR.mkdir(parents=True, exist_ok=True)
        datasets = {
            p.stem.rsplit("_", 1)[0]
            for p in config.FEATURE_SELECTED_DIR.glob("*_train.parquet")
        }
        for ds in datasets:
            path = config.FEATURE_SELECTED_DIR / f"{ds}_train.parquet"
            df = pd.read_parquet(path)
            X = df.drop(columns=DEFAULT_TARGETS + OTTER_META_COLS, errors="ignore")
            comps = reduce_pca(X, self.n_components)
            out = config.DIM_REDUCED_DIR / f"{ds}_train.parquet"
            comps.to_parquet(out, index=False)
            print(f"[REPORT] {ds} PCA shape={comps.shape}")
        print("[SUCCESS] dimensionality reduction completed.")
        return 0
