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
import matplotlib.pyplot as plt
import seaborn as sns

from FIT_python.Visualisations.plot_style import apply_style
from FIT_python.caption_utils import save_caption

from FIT_python import config
from FIT_python.data_split_and_summary.datasplit_and_summary_wraper import SplitWrapper
from FIT_python.data_split_and_summary.summary_data_utils import compute_summary
from FIT_python.pipeline_individual_id.evaluation import (
    sequential_holdout_ids,
    compute_overlap_jsl_style_vec,
)
from FIT_python.pipeline_individual_id.generate_trails_and_trailpairs import (
    generate_pairwise_comparisons_from_df,
)
from FIT_python.pipeline_individual_id import (
    geometric_pairwise_projection,
    pairwise_individual_id_pipeline,
)
from FIT_python.pipeline_sex.sex_predict_and_visualisation import predict_all
from FIT_python.utils import get_species_paths


def _ensure_splits(species: str) -> Path:
    """Return directory containing train/test/inference splits."""
    splits_dir = config.SPLITS_DIR / species
    if not splits_dir.exists() or not (splits_dir / "train.parquet").exists():
        print(f"[INFO] Splits for {species!r} not found – creating via SplitWrapper")
        SplitWrapper().split_all(reuse_splits=True)
    return splits_dir


def _load_splits(species: str) -> dict[str, pd.DataFrame]:
    """Load train and test dataframes and print a short summary."""
    splits_dir = _ensure_splits(species)
    dfs = {}
    # only consider the train and test splits for model development and
    # evaluation. The optional inference split is ignored here on purpose.
    for name in ["train", "test"]:
        fp = splits_dir / f"{name}.parquet"
        if fp.exists():
            df = pd.read_parquet(fp)
            if "Fold" in df.columns and "fold" not in df.columns:
                df = df.rename(columns={"Fold": "fold"})
            if "trail" in df.columns and "Trail" not in df.columns:
                df = df.rename(columns={"trail": "Trail"})
            dfs[name] = df
            compute_summary(df, species, name)
    return dfs


def _select_features(dfs: dict[str, pd.DataFrame]) -> List[str]:
    """Return the names of all morphometric feature columns."""
    df_train = dfs["train"]
    morph_cols = [
        c
        for c in df_train.columns
        if c.startswith(("dist", "ang", "t", "v"))
        and c != "trail"
        and df_train[c].dtype.kind in "if"
    ]
    print(f"[INFO] morphometric columns: {morph_cols}")

    return morph_cols


def _add_sex_predictions(species: str, dfs: dict[str, pd.DataFrame]) -> None:
    """Append sex model predictions as additional features."""
    pred_df = predict_all(
        species,
        prefer_generic=True,
        models_dir=get_species_paths(species)["search"],
        use_cv_train_predictions=True,
    )
    sex_cols = [c for c in pred_df.columns if c.startswith("pred_")]
    for name, df in dfs.items():
        sub = pred_df[pred_df["__split__"] == name]
        if not sub.empty:
            dfs[name] = pd.concat([df.reset_index(drop=True), sub[sex_cols].reset_index(drop=True)], axis=1)


def scale_holdout_split(
    train_df: pd.DataFrame, val_df: pd.DataFrame, morph_cols: List[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Scale ``morph_cols`` in ``train_df`` and ``val_df`` using ``StandardScaler``.

    The scaler is fitted on the training dataframe and then applied to both
    the training and validation dataframes. Copies of the inputs with the
    transformed columns are returned.
    """

    scaler = StandardScaler().fit(train_df[morph_cols])

    train_scaled = train_df.copy()
    val_scaled = val_df.copy()

    train_scaled[morph_cols] = scaler.transform(train_df[morph_cols])
    val_scaled[morph_cols] = scaler.transform(val_df[morph_cols])

    return train_scaled, val_scaled


def _confusion_from_results(results: List[Dict]) -> list[list[int]]:
    df = pd.DataFrame(results)
    if df.empty:
        return [[0, 0], [0, 0]]
    df["pred_same"] = compute_overlap_jsl_style_vec(df)
    y_true = df["same_individual"].astype(str) == "True"
    y_pred = df["pred_same"]
    cm = confusion_matrix(y_true, y_pred, labels=[True, False])
    return cm.tolist()


def main(species: str = "eurasian_otter") -> None:
    dfs_raw = _load_splits(species)
    if not dfs_raw:
        raise RuntimeError(f"No split data available for {species}")

    _add_sex_predictions(species, dfs_raw)
    morph_cols = _select_features(dfs_raw)

    unique_ids = dfs_raw["train"]["individual_id"].dropna().unique()
    splits = sequential_holdout_ids(
        unique_ids, val_sizes=[2, 4, 6, 8], n_iter=1, random_state=config.GLOBAL_RANDOM_SEED
    )

    out_dir = config.PATHS["individual_id"] / "tables" / "individual_id_pipelines"
    out_dir.mkdir(parents=True, exist_ok=True)

    for split_idx, split in enumerate(splits):
        train_ids = split["train_ids"]
        val_ids = split["val_ids"]
        df_train = dfs_raw["train"].loc[dfs_raw["train"].individual_id.isin(train_ids)]
        df_val = dfs_raw["train"].loc[dfs_raw["train"].individual_id.isin(val_ids)]

        df_train, df_val = scale_holdout_split(df_train, df_val, morph_cols)

        # Optional correlation heatmap for the scaled morphometric features
        corr = df_train[morph_cols].corr()
        apply_style()
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(corr, cmap="viridis", center=0, ax=ax)
        ax.set_xlabel("Morphometric Features")
        ax.set_ylabel("Morphometric Features")
        fig.tight_layout()
        fig_dir = config.PATHS["individual_id"] / "figures"
        fig_dir.mkdir(parents=True, exist_ok=True)
        out_file = fig_dir / f"morph_corr_heatmap_split_{split_idx}.png"
        fig.savefig(out_file)
        save_caption(out_file, "Correlation between scaled morphometric features")
        plt.close(fig)

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
            sexmodel_path=str(
                get_species_paths(species)["models"]
                / config.SEX_PREDICT_METRIC
                / f"{species}.joblib"
            ),
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
