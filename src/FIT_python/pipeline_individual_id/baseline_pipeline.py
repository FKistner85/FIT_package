"""Distance baseline pipeline for individual ID.

This module provides a thin wrapper around
`run_all_pairwise_projections_parallel` to evaluate
validation trail pairs using a reference control
variation (RCV) set constructed from the training
samples.
"""

from __future__ import annotations

from typing import Iterable, List, Dict, Optional
from pathlib import Path

import pandas as pd

from .pairwise_individual_id_pipeline import run_all_pairwise_projections_parallel
from FIT_python.config import CONFIG
from FIT_python.utils import get_species_paths
from FIT_python.config import SEX_PREDICT_METRIC


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
        outlier_methods: Optional[List[str] | str] = CONFIG["pipeline_individual_id"][
            "pairwise_defaults"
        ]["outlier_methods"],
        scaler_methods: Optional[List[str] | str] = CONFIG["pipeline_individual_id"][
            "pairwise_defaults"
        ]["scaler_methods"],
        use_sexmodel_prediction: bool = CONFIG["pipeline_individual_id"]["pairwise_defaults"][
            "use_sexmodel_prediction"
        ],
        sexmodel_path: str | None = None,
        n_jobs: int = -1,
        debug: bool = False,
        tag: str | None = None,
        fold: int | str | None = None,
        master_fp: Path | None = None,
        origin: str = "test",
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
            ``get_species_paths(species)['models']/SEX_PREDICT_METRIC/{species}.joblib``.
        n_jobs:
            Number of parallel jobs for the underlying pipeline.
        debug:
            Forwarded to the projection function for verbose output.
        tag, fold, master_fp, origin:
            When ``master_fp`` is provided, the resulting DataFrame is annotated
            with ``origin``/``tag``/``fold`` and appended to this parquet file.
        """
        df_rcv = train_df.copy()
        if "individual_id" in df_rcv.columns:
            df_rcv["individual_id"] = "RCV"
        if "trail" in df_rcv.columns:
            df_rcv["trail"] = "RCV"
        if "Trail" in df_rcv.columns:
            df_rcv["Trail"] = "RCV"

        base_df = df_rcv if val_df is None else pd.concat([df_rcv, val_df], ignore_index=True)

        from FIT_python import config as cfg
        import warnings

        model_fp = sexmodel_path

        if use_sexmodel_prediction and model_fp is None:
            if "species" not in base_df.columns:
                raise ValueError(
                    "sexmodel_path must be provided when use_sexmodel_prediction=True"
                )

            # Spezies normalisieren
            species_values = base_df["species"].dropna().unique()
            if len(species_values) == 0:
                raise ValueError("No species found in dataframe for sexmodel lookup")
            species = str(species_values[0]).lower().replace(" ", "_")

            # Mapping anwenden
            mapped_species = cfg.SPECIES_MODEL_MAP.get(species, species)
            if mapped_species != species:
                warnings.warn(f"Mapped species '{species}' → '{mapped_species}'", stacklevel=2)
            species = mapped_species

            # Immer Bayes-Ordner nutzen, Metric soft-coded
            metric_folder = CONFIG["pipeline_individual_id"]["pairwise_defaults"].get(
                "metric_folder", SEX_PREDICT_METRIC
            )
            paths = get_species_paths(species)
            model_fp = paths["models"] / metric_folder / f"{species}.joblib"

            if not model_fp.is_file():
                raise FileNotFoundError(f"Sex model file not found: {model_fp}")

        results = run_all_pairwise_projections_parallel(
            list(val_comparisons),
            base_df,
            feature_cols,
            k_features=CONFIG["pipeline_individual_id"]["pairwise_defaults"]["k_features"],
            reducers=reducers or ["lda"],
            selection_method=selection_method,
            n_components=n_components,
            outlier_methods=outlier_methods,
            scaler_methods=scaler_methods,
            use_sexmodel_prediction=use_sexmodel_prediction,
            sexmodel_path=model_fp,
            debug=debug,
            n_jobs=n_jobs,
            fit_df=df_rcv,
        )

        df_res = pd.DataFrame(results)
        if master_fp is not None:
            df_mp = df_res.copy()
            df_mp["origin"] = origin
            df_mp["tag"] = tag
            df_mp["fold"] = fold
            df_mp.to_parquet(master_fp, append=True)

        return df_res
