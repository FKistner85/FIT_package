"""Distance baseline pipeline for individual ID.

This module provides a thin wrapper around
`run_all_pairwise_projections_parallel` to evaluate
validation trail pairs using a reference control
variation (RCV) set constructed from the training
samples.
"""

from __future__ import annotations

from typing import Iterable, List, Dict, Optional

import pandas as pd

from .pairwise_individual_id_pipeline import run_all_pairwise_projections_parallel
from FIT_python.soft_config import SOFT_CONFIG
from FIT_python.config import PATHS


class DistanceBaseline:
    """Utility class to compute baseline distances on validation comparisons."""

    @staticmethod
    def run(
        train_df: pd.DataFrame,
        val_comparisons: Iterable[Dict],
        feature_cols: List[str],
        *,
        val_df: pd.DataFrame | None = None,
        reducers: List[str] | None = None,
        selection_method: str = "forward",
        n_components: int | List[int] = 2,
        outlier_methods: Optional[List[str] | str] = SOFT_CONFIG["pipeline_individual_id"][
            "pairwise_defaults"
        ]["outlier_methods"],
        scaler_methods: Optional[List[str] | str] = SOFT_CONFIG["pipeline_individual_id"][
            "pairwise_defaults"
        ]["scaler_methods"],
        use_sexmodel_prediction: bool = SOFT_CONFIG["pipeline_individual_id"]["pairwise_defaults"][
            "use_sexmodel_prediction"
        ],
        sexmodel_path: str | None = None,
        n_jobs: int = -1,
        debug: bool = False,
    ) -> pd.DataFrame:
        """Run the distance baseline on the given comparisons.

        Parameters
        ----------
        train_df:
            DataFrame containing the training data. All rows will be used as the
            RCV set by replacing ``individual_id`` and ``trail`` with ``"RCV"``.
        val_comparisons:
            List of comparison dictionaries as produced by
            ``generate_pairwise_comparisons_from_df``.
        feature_cols:
            Names of the morphometric feature columns to use.
        val_df:
            Optional validation dataframe containing the samples referenced in
            ``val_comparisons``. If provided, it will be concatenated to the
            training dataframe so that all required IDs are present.
        reducers:
            Dimensionality reduction methods passed through to
            ``run_all_pairwise_projections_parallel``. Defaults to ``["lda"]``.
        selection_method:
            Feature selection strategy, defaults to ``"forward"``.
        n_components:
            Target dimensionality for the reducers.
        outlier_methods, scaler_methods:
            Optional preprocessing methods.
        use_sexmodel_prediction:
            Whether to append sex-model probabilities as features.
        sexmodel_path:
            Path to the sex classifier. If ``use_sexmodel_prediction`` is
            ``True`` and no path is given, the classifier location is derived
            from the ``species`` column via
            ``PATHS['random_search']/best_balanced_test_acc/<species>.joblib``.
        n_jobs:
            Number of parallel jobs for the underlying pipeline.
        debug:
            Forwarded to the projection function for verbose output.
        """
        df_rcv = train_df.copy()
        if "individual_id" in df_rcv.columns:
            df_rcv["individual_id"] = "RCV"
        if "trail" in df_rcv.columns:
            df_rcv["trail"] = "RCV"
        if "Trail" in df_rcv.columns:
            df_rcv["Trail"] = "RCV"

        base_df = df_rcv if val_df is None else pd.concat([df_rcv, val_df], ignore_index=True)

        model_fp = sexmodel_path
        if use_sexmodel_prediction and model_fp is None:
            if "species" not in base_df.columns:
                raise ValueError(
                    "sexmodel_path must be provided when use_sexmodel_prediction=True"
                )
            species = str(base_df["species"].dropna().unique()[0])
            model_fp = (
                PATHS["random_search"]
                / "best_balanced_test_acc"
                / f"{species}.joblib"
            )

        results = run_all_pairwise_projections_parallel(
            list(val_comparisons),
            base_df,
            feature_cols,
            train_df=df_rcv,
            k_features=SOFT_CONFIG["pipeline_individual_id"]["pairwise_defaults"]["k_features"],
            reducers=reducers or ["lda"],
            selection_method=selection_method,
            n_components=n_components,
            outlier_methods=outlier_methods,
            scaler_methods=scaler_methods,
            use_sexmodel_prediction=use_sexmodel_prediction,
            sexmodel_path=model_fp,
            debug=debug,
            n_jobs=n_jobs,
        )

        return pd.DataFrame(results)
