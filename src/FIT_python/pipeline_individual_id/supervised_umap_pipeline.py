# Pipeline for supervised UMAP based individual identification

from __future__ import annotations

from typing import List, Sequence, Optional, Tuple, Dict

from FIT_python.data_split_and_summary.data_import_utils import coerce_numeric_columns

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from ..general_pipeline_steps.outlier_wrapper import OutlierCleanerTransformer
from ..general_pipeline_steps.feature_selection_wrapper import (
    FeatureSelectionTransformer,
)
from ..general_pipeline_steps.feature_scaler_wrapper import FeatureScalerTransformer
from ..general_pipeline_steps.dimensionality_reduction_wrapper import (
    DimensionalityReducerTransformer,
)
from .distance_metrics import compute_distances


DistanceRecord = Dict[str, float | str | bool]


def _prepare_features(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    sex_features: Sequence[str] | None,
    *,
    outlier_method: str | None,
    scaler_method: str | None,
    selection_method: str,
    k_features: int,
    random_state: int,
) -> Tuple[pd.DataFrame, Pipeline]:
    """Return processed feature matrix and the fitted preprocessing pipeline."""

    df = coerce_numeric_columns(df.copy())
    df[feature_cols] = df[feature_cols].apply(pd.to_numeric, errors="coerce")
    X = df[feature_cols].astype(float)
    if X.select_dtypes(exclude=[float, int]).shape[1] > 0:
        bad_cols = X.select_dtypes(exclude=[float, int]).columns.tolist()
        raise ValueError(f"Non-numeric data found in columns: {bad_cols}")
    y = df.get("individual_id")

    steps: list[tuple[str, object]] = []
    if outlier_method:
        steps.append(("outlier", OutlierCleanerTransformer(method=outlier_method)))
    selector = FeatureSelectionTransformer(
        method=selection_method, k=k_features, random_state=random_state
    )
    steps.append(("select", selector))
    if scaler_method:
        steps.append(("scale", FeatureScalerTransformer(method=scaler_method)))

    pipe = Pipeline(steps)
    X_prep = pipe.fit_transform(X, y)

    if not isinstance(X_prep, pd.DataFrame):
        cols = selector.get_feature_names_out()
        X_prep = pd.DataFrame(X_prep, index=df.index, columns=cols)

    if sex_features:
        sex_df = df[list(sex_features)].copy()
        for col in sex_df.columns:
            if sex_df[col].dtype == object:
                sex_df[col] = sex_df[col].astype(str).str.lower().map({"f": 0, "m": 1})
            sex_df[col] = pd.to_numeric(sex_df[col], errors="coerce")
        X_prep = pd.concat([X_prep, sex_df], axis=1)

    return X_prep, pipe


def _compute_pair_features(
    embeddings: pd.DataFrame,
    comparisons: Sequence[dict],
) -> pd.DataFrame:
    """Compute distance features for each comparison.

    The ``embeddings`` DataFrame is indexed by ``id`` so ``samples_a`` and
    ``samples_b`` can be looked up directly via ``loc``.
    """

    records: list[DistanceRecord] = []
    for comp in comparisons:

        ids_a = [str(i) for i in comp["samples_a"]]
        ids_b = [str(i) for i in comp["samples_b"]]
        ca = embeddings.loc[ids_a].to_numpy().mean(axis=0)
        cb = embeddings.loc[ids_b].to_numpy().mean(axis=0)

        dists = compute_distances(ca, cb)
        rec: DistanceRecord = {
            "trail_a_id": comp["trail_a_id"],
            "trail_b_id": comp["trail_b_id"],
            "same_individual": bool(comp.get("same_individual", False)),
        }
        for key, val in dists.items():
            rec[f"dist_{key}"] = float(val)
        records.append(rec)

    return pd.DataFrame(records)


def run(
    train_df: pd.DataFrame,
    val_comparisons: Sequence[dict],
    feature_cols: Sequence[str],
    sex_features: Sequence[str] | None = None,
    *,
    outlier_method: str | None = None,
    scaler_method: str | None = None,
    selection_method: str = "forward",
    k_features: int = 15,
    dist_selection_method: str | None = None,
    k_distances: int | None = None,
    classifier=None,
    random_state: int = 0,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Fit a supervised UMAP pipeline and classify validation pairs.

    Parameters
    ----------
    train_df : pd.DataFrame
        Training footprints with ``individual_id`` and feature columns.
    val_comparisons : sequence of dict
        Pair specifications containing ``samples_a``/``samples_b`` lists of
        ``id`` values and ``same_individual`` label.
    feature_cols : sequence of str
        Columns used as numeric input features.
    sex_features : sequence of str, optional
        Extra columns appended before dimensionality reduction.

    Returns
    -------
    tuple of (predictions_df, embeddings_df)
    """

    X_prep, prep_pipe = _prepare_features(
        train_df,
        feature_cols,
        sex_features,
        outlier_method=outlier_method,
        scaler_method=scaler_method,
        selection_method=selection_method,
        k_features=k_features,
        random_state=random_state,
    )

    reducer = DimensionalityReducerTransformer(
        method="umap",
        n_components=2,
        supervised=True,
        random_state=random_state,
    )
    labels = train_df.get("individual_id")
    if not np.issubdtype(labels.dtype, np.number):
        labels = pd.factorize(labels)[0]
    embeddings = reducer.fit_transform(X_prep, labels)
    emb_df = pd.DataFrame(
        embeddings,
        index=train_df["id"].astype(str),
        columns=reducer.get_feature_names_out(),
    )

    dist_df = _compute_pair_features(emb_df, val_comparisons)

    X_dist = dist_df[[c for c in dist_df.columns if c.startswith("dist_")]]
    y_dist = dist_df["same_individual"].astype(int)

    if dist_selection_method:
        dist_selector = FeatureSelectionTransformer(
            method=dist_selection_method,
            k=k_distances,
            random_state=random_state,
        )
        X_dist_sel = dist_selector.fit_transform(X_dist, y_dist)
        if not isinstance(X_dist_sel, pd.DataFrame):
            sel_names = dist_selector.get_feature_names_out()
            X_dist_sel = pd.DataFrame(X_dist_sel, columns=sel_names)
    else:
        X_dist_sel = X_dist

    if classifier is None:
        classifier = LogisticRegression(max_iter=200)

    classifier.fit(X_dist_sel, y_dist)
    proba = classifier.predict_proba(X_dist_sel)[:, 1]
    dist_df = dist_df.copy()
    dist_df["pred_same_proba"] = proba

    return dist_df, emb_df
