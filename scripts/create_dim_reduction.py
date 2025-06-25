#!/usr/bin/env python3
"""Apply PCA on feature selected datasets."""

import pandas as pd
from sklearn.decomposition import PCA

from FIT_python.config import (
    FEATURE_SELECTED_DIR,
    DIM_REDUCED_DIR,
    DEFAULT_TARGETS,
    OTTER_META_COLS,
    DEBUG_MODE,
)
from FIT_python.path_utils import feature_selected_path, dim_reduced_path

N_COMPONENTS = 2


def main() -> None:
    if not FEATURE_SELECTED_DIR.exists():
        msg = f"Required file not found: {FEATURE_SELECTED_DIR}"
        if DEBUG_MODE:
            raise FileNotFoundError(msg)
        else:
            print("Skipping:", msg)
            return

    DIM_REDUCED_DIR.mkdir(parents=True, exist_ok=True)
    datasets = set(p.stem.rsplit("_", 1)[0] for p in FEATURE_SELECTED_DIR.glob("*_train.parquet"))

    for dataset in datasets:
        train_p = feature_selected_path(dataset, "train")
        if not train_p.exists():
            msg = f"Required file not found: {train_p}"
            if DEBUG_MODE:
                raise FileNotFoundError(msg)
            else:
                print("Skipping:", msg)
                continue

        df_train = pd.read_parquet(train_p)
        meta_cols = OTTER_META_COLS if "otter" in dataset.lower() else ["id"]
        feature_cols = [c for c in df_train.columns if c not in meta_cols + DEFAULT_TARGETS]

        pca = PCA(n_components=N_COMPONENTS, random_state=0)
        comps = pca.fit_transform(df_train[feature_cols])
        cols = [f"PC{i+1}" for i in range(N_COMPONENTS)]
        df_out = pd.DataFrame(comps, columns=cols)
        df_out.to_parquet(dim_reduced_path(dataset, "train"), index=False)
        print(f"Dimensionality reduced for {dataset}")


if __name__ == "__main__":
    main()
