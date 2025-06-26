# src/FIT_python/dim_reduction_wrapper.py
"""Wrapper to apply dimensionality reduction on multiple datasets."""

from pathlib import Path
from typing import Dict, List, Optional
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
    """Apply dimensionality reduction on feature-selected datasets."""

    def __init__(
        self,
        feature_dir: Path | None = None,
        out_dir: Path | None = None,
        methods: Optional[List[str]] = None,
        n_components: int | None = None,
        params: Optional[Dict[str, dict]] = None,
    ) -> None:
        self.feature_dir = feature_dir or config.FEATURE_SELECTED_DIR
        self.out_dir = out_dir or config.DIM_REDUCED_DIR
        self.methods = methods or getattr(config, "DIM_REDUCTION_METHODS", ["pca"])
        self.n_components = n_components or getattr(config, "N_COMPONENTS", 2)
        self.params = params or getattr(config, "DIM_PARAMS", {})

    def reduce_all(self) -> int:
        if not self.feature_dir.exists():
            msg = f"Required directory not found: {self.feature_dir}"
            print(f"[ERROR] {msg}", file=sys.stderr)
            if config.DEBUG_MODE:
                raise FileNotFoundError(msg)
            return 1
        self.out_dir.mkdir(parents=True, exist_ok=True)

        for method_dir in sorted(self.feature_dir.iterdir()):
            if not method_dir.is_dir():
                continue
            method_name = method_dir.name
            datasets = {
                p.stem.rsplit("_", 1)[0] for p in method_dir.glob("*_train.parquet")
            }
            for ds in datasets:
                df_train = pd.read_parquet(method_dir / f"{ds}_train.parquet")
                df_test = pd.read_parquet(method_dir / f"{ds}_test.parquet")
                meta_cols = OTTER_META_COLS if "otter" in ds.lower() else ["id"]
                X_train = df_train.drop(columns=meta_cols + DEFAULT_TARGETS)
                X_test = df_test.drop(columns=meta_cols + DEFAULT_TARGETS)
                for dm in self.methods:
                    func = METHOD_MAP.get(dm)
                    if func is None:
                        raise ValueError(f"Unknown method '{dm}'")
                    out_base = self.out_dir / dm / method_name
                    out_base.mkdir(parents=True, exist_ok=True)
                    method_params = {"n_components": self.n_components}
                    method_params.update(self.params.get(dm, {}))
                    try:
                        red_train = func(X_train, **method_params)
                        red_test = func(X_test, **method_params)
                    except Exception as exc:  # pragma: no cover - optional methods may fail
                        print(f"[WARN] {dm} failed on {ds}: {exc}")
                        continue
                    pd.concat([df_train[meta_cols + DEFAULT_TARGETS], red_train], axis=1).to_parquet(
                        out_base / f"{ds}_train.parquet", index=False
                    )
                    pd.concat([df_test[meta_cols + DEFAULT_TARGETS], red_test], axis=1).to_parquet(
                        out_base / f"{ds}_test.parquet", index=False
                    )
                    print(
                        f"[REPORT] {ds} {method_name}->{dm} train shape={red_train.shape}"
                    )

                # backward compatible PCA file without subdirectories
                default_out = self.out_dir / f"{ds}_train.parquet"
                if not default_out.exists():
                    default_train = reduce_pca(X_train, self.n_components, random_state=config.GLOBAL_RANDOM_SEED)
                    default_train.to_parquet(default_out, index=False)
        print("[SUCCESS] dimensionality reduction completed.")
        return 0
