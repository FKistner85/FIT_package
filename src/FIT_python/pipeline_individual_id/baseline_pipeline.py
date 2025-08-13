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
import time

import pandas as pd

from .pairwise_individual_id_pipeline import run_all_pairwise_projections_parallel
from .generate_trails_and_trailpairs import generate_pairwise_comparisons_from_df
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
            ``get_species_paths(species, section='individual_id')['models']/SEX_PREDICT_METRIC/{species}.joblib``.
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
            paths = get_species_paths(species, section="individual_id")
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


def evaluate_test_and_inference(
    train_df: pd.DataFrame,
    df_test: pd.DataFrame,
    feature_cols: List[str],
    *,
    df_inf: pd.DataFrame | None = None,
    include_inference: bool = False,
    master_pairs: list[pd.DataFrame] | None = None,
    all_master_pairs: list[pd.DataFrame] | None = None,
    species: str | None = None,
    tag: str | None = None,
    reducers: List[str] | None = None,
    selection_method: str = "forward",
    n_components: int | List[int] = 2,
    outlier_methods: Optional[List[str] | str] = CONFIG["pipeline_individual_id"][
        "pairwise_defaults"
    ]["outlier_methods"],
    scaler_methods: Optional[List[str] | str] = CONFIG["pipeline_individual_id"][
        "pairwise_defaults"
    ]["scaler_methods"],
    use_sexmodel_prediction: bool = CONFIG["pipeline_individual_id"][
        "pairwise_defaults"
    ]["use_sexmodel_prediction"],
    sexmodel_path: str | None = None,
    n_jobs: int = -1,
    master_fp: Path | None = None,
) -> tuple[list[pd.DataFrame], list[pd.DataFrame]]:
    """Evaluate DistanceBaseline on test and optionally inference pairs.

    Parameters
    ----------
    train_df, df_test:
        Training and test dataframes.
    feature_cols:
        Names of the morphometric feature columns to use.
    df_inf:
        Optional inference dataframe used when ``include_inference`` is ``True``.
    include_inference:
        When ``True`` inference pairs are generated and evaluated.
    master_pairs, all_master_pairs:
        Lists that will receive the per-origin results. New dataframes are
        appended to these lists when provided.

    Returns
    -------
    tuple[list[pd.DataFrame], list[pd.DataFrame]]
        Updated ``master_pairs`` and ``all_master_pairs`` lists.
    """

    if master_pairs is None:
        master_pairs = []
    if all_master_pairs is None:
        all_master_pairs = []

    # --- Test pairs ---
    if not df_test.empty:
        comps_test = generate_pairwise_comparisons_from_df(df_test, trail_col="Trail")
        start = time.time()
        res_test = DistanceBaseline.run(
            train_df,
            comps_test,
            feature_cols,
            val_df=df_test,
            reducers=reducers,
            selection_method=selection_method,
            n_components=n_components,
            outlier_methods=outlier_methods,
            scaler_methods=scaler_methods,
            use_sexmodel_prediction=use_sexmodel_prediction,
            sexmodel_path=sexmodel_path,
            n_jobs=n_jobs,
            tag=tag,
            master_fp=master_fp,
            origin="test",
        )
        res_test["origin"] = "test"
        if species is not None:
            res_test["species"] = species
        if tag is not None:
            res_test["tag"] = tag
        res_test["runtime_s"] = time.time() - start
        master_pairs.append(res_test)
        all_master_pairs.append(res_test)

    # --- Inference pairs ---
    if include_inference and df_inf is not None and not df_inf.empty:
        comps_inf = generate_pairwise_comparisons_from_df(df_inf, trail_col="Trail")
        start = time.time()
        res_inf = DistanceBaseline.run(
            train_df,
            comps_inf,
            feature_cols,
            val_df=df_inf,
            reducers=reducers,
            selection_method=selection_method,
            n_components=n_components,
            outlier_methods=outlier_methods,
            scaler_methods=scaler_methods,
            use_sexmodel_prediction=use_sexmodel_prediction,
            sexmodel_path=sexmodel_path,
            n_jobs=n_jobs,
            tag=tag,
            master_fp=master_fp,
            origin="inference",
        )
        res_inf["origin"] = "inference"
        if species is not None:
            res_inf["species"] = species
        if tag is not None:
            res_inf["tag"] = tag
        res_inf["runtime_s"] = time.time() - start
        master_pairs.append(res_inf)
        all_master_pairs.append(res_inf)

    return master_pairs, all_master_pairs
