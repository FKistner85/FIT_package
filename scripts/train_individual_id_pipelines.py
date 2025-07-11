#!/usr/bin/env python
"""Train and evaluate individual ID pipelines.

This script orchestrates the three pipelines described in
``docs/pipeline_individual_id_three_pipelines.md``.  It reuses the
train/test splits from the sex classification pipeline, adds scaled
morphometric features and sex model predictions, generates sequential
holdout splits and runs each pipeline on the resulting trail pairs.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Dict

import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler

from FIT_python import config
from FIT_python.data_split_and_summary.split_wrapper import SplitWrapper
from FIT_python.data_split_and_summary.summary_data_utils import compute_summary
from FIT_python.pipeline_individual_id.utils import (
    sequential_holdout_ids,
    compute_overlap_jsl_style,
)
from FIT_python.pipeline_individual_id.generate_trails_and_trailpairs import (
    generate_pairwise_comparisons_from_df,
)
from FIT_python.pipeline_individual_id import (
    geometric_pairwise_projection,
    pairwise_individual_id_pipeline,
)
from FIT_python.pipeline_sex.sex_predict_and_visualisation import predict_all


def _ensure_splits(species: str) -> Path:
    """Return directory containing train/test/inference splits."""
    splits_dir = config.SPLITS_DIR / species
    if not splits_dir.exists() or not (splits_dir / "train.parquet").exists():
        print(f"[INFO] Splits for {species!r} not found – creating via SplitWrapper")
        SplitWrapper().split_all()
    return splits_dir


def _load_splits(species: str) -> dict[str, pd.DataFrame]:
    """Load train/test/inference dataframes and print a short summary."""
    splits_dir = _ensure_splits(species)
    dfs = {}
    for name in ["train", "test", "inference"]:
        fp = splits_dir / f"{name}.parquet"
        if fp.exists():
            df = pd.read_parquet(fp)
            dfs[name] = df
            compute_summary(df, species, name)
    return dfs


def _select_and_scale_features(dfs: dict[str, pd.DataFrame]) -> tuple[List[str], dict[str, pd.DataFrame]]:
    """Return morphometric columns and scaled copies of the splits."""
    df_train = dfs["train"]
    morph_cols = [
        c
        for c in df_train.columns
        if c.startswith(("dist", "ang", "t", "v"))
        and c != "trail"
        and df_train[c].dtype.kind in "if"
    ]
    print(f"[INFO] morphometric columns: {morph_cols}")

    scaler = StandardScaler().fit(df_train[morph_cols])
    scaled = {}
    for name, df in dfs.items():
        df2 = df.copy()
        df2[morph_cols] = scaler.transform(df2[morph_cols])
        scaled[name] = df2
    return morph_cols, scaled


def _add_sex_predictions(species: str, dfs: dict[str, pd.DataFrame]) -> None:
    """Append sex model predictions as additional features."""
    pred_df = predict_all(species, prefer_generic=True)
    sex_cols = [c for c in pred_df.columns if c.startswith("pred_")]
    for name, df in dfs.items():
        sub = pred_df[pred_df["__split__"] == name]
        if not sub.empty:
            dfs[name] = pd.concat([df.reset_index(drop=True), sub[sex_cols].reset_index(drop=True)], axis=1)


def _confusion_from_results(results: List[Dict]) -> list[list[int]]:
    df = pd.DataFrame(results)
    if df.empty:
        return [[0, 0], [0, 0]]
    df["pred_same"] = df.apply(compute_overlap_jsl_style, axis=1)
    y_true = df["same_individual"].astype(str) == "True"
    y_pred = df["pred_same"]
    cm = confusion_matrix(y_true, y_pred, labels=[True, False])
    return cm.tolist()


def main(species: str = "eurasian_otter") -> None:
    dfs_raw = _load_splits(species)
    if not dfs_raw:
        raise RuntimeError(f"No split data available for {species}")

    morph_cols, dfs = _select_and_scale_features(dfs_raw)
    _add_sex_predictions(species, dfs)

    unique_ids = dfs["train"]["individual_id"].dropna().unique()
    splits = sequential_holdout_ids(unique_ids, val_sizes=[2, 4, 6, 8], n_iter=1, random_state=0)

    out_dir = config.RESULTS_DATA_DIR / "individual_id_pipelines"
    out_dir.mkdir(parents=True, exist_ok=True)

    for split_idx, split in enumerate(splits):
        train_ids = split["train_ids"]
        val_ids = split["val_ids"]
        df_train = dfs["train"].loc[dfs["train"].individual_id.isin(train_ids)]
        df_val = dfs["train"].loc[dfs["train"].individual_id.isin(val_ids)]

        comps, _ = generate_pairwise_comparisons_from_df(df_val)

        results = {}

        res1 = geometric_pairwise_projection.run_all_pairwise_projections_parallel(
            comps,
            pd.concat([df_train, df_val], ignore_index=True),
            feature_cols=morph_cols,
            k_features=16,
            reducers=["pca"],
            n_components=2,
            use_sexmodel_prediction=False,
            n_jobs=1,
        )
        results["geometric"] = _confusion_from_results(res1)

        res2 = pairwise_individual_id_pipeline.run_all_pairwise_projections_parallel(
            comps,
            pd.concat([df_train, df_val], ignore_index=True),
            feature_cols=morph_cols,
            k_features=100,
            reducers=["umap"],
            n_components=2,
            use_sexmodel_prediction=True,
            sexmodel_path=str(config.RESULTS_DATA_DIR / "random_search_standard_metrics" / "best_balanced_test_acc" / f"{species}.joblib"),
            n_jobs=1,
        )
        results["umap"] = _confusion_from_results(res2)

        res3 = pairwise_individual_id_pipeline.run_embedding_once_pipeline(
            comps,
            pd.concat([df_train, df_val], ignore_index=True),
            feature_cols=morph_cols,
            k_features=100,
            reducer="lda",
            n_components=2,
        )
        results["siamese"] = _confusion_from_results(res3)

        with open(out_dir / f"split_{split_idx}.json", "w") as fh:
            json.dump(results, fh, indent=2)


if __name__ == "__main__":
    main()
