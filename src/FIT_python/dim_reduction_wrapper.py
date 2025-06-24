# src/FIT_python/dim_reduction_wrapper.py
"""Wrapper to apply dimensionality reduction on multiple datasets."""

from pathlib import Path
from typing import Dict, List
import pandas as pd

from FIT_python.config import PROCESSED_DIR
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
