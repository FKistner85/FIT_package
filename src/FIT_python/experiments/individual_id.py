from __future__ import annotations

from pathlib import Path
import itertools
import time
from typing import Iterable

import pandas as pd
from tqdm import tqdm
from tqdm_joblib import tqdm_joblib

from FIT_python.config import SPLITS_DIR, RESULTS_DATA_DIR
from FIT_python.pipeline_individual_id.pairwise_individual_id_pipeline import (
    run_all_pairwise_projections_parallel,
)
from FIT_python.pipeline_individual_id.generate_trails_and_trailpairs import (
    generate_pairwise_comparisons_from_df,
)


_DEFAULT_FEATURE_COUNTS = [2, 5, 10, 15, 20, 50, 100, 200]
_SELECTION_METHODS = ["random_forest", "forward", "univariate", "variance", "lasso"]
_REDUCERS = ["pca", "lda", "umap"]
_N_COMPONENTS_LIST = [2, 3, 5, 10]
_OUTLIER_METHODS = [None, "clip", "zscore"]
_SCALER_METHODS = [None, "standard", "robust"]
_USE_SEX_OPTS = [False, True]
_BASELINE_COUNTS = [10, 12, 14, 16, 18, 20]


def _default_grid(random_state: int = 42, n_samples: int = 2) -> pd.DataFrame:
    """Return the default hyper-parameter grid with some random samples."""

    grid = pd.DataFrame(
        itertools.product(
            _DEFAULT_FEATURE_COUNTS,
            _SELECTION_METHODS,
            _REDUCERS,
            _N_COMPONENTS_LIST,
            _OUTLIER_METHODS,
            _SCALER_METHODS,
            _USE_SEX_OPTS,
        ),
        columns=[
            "feature_count",
            "selection_method",
            "reducer",
            "n_components",
            "outlier_method",
            "scaler_method",
            "use_sex",
        ],
    ).sort_values("feature_count", ascending=False).reset_index(drop=True)

    sampled = grid.sample(n=n_samples, random_state=random_state)
    sampled = sampled.sort_values("feature_count", ascending=False).reset_index(drop=True)

    extra = []
    for fc in _BASELINE_COUNTS:
        for use_sex in [False, True]:
            extra.append(
                {
                    "feature_count": fc,
                    "selection_method": "forward",
                    "reducer": "lda",
                    "n_components": 2,
                    "outlier_method": None,
                    "scaler_method": None,
                    "use_sex": use_sex,
                }
            )

    extra_df = pd.DataFrame(extra)
    sampled = pd.concat([sampled, extra_df], ignore_index=True)
    return sampled.sort_values("feature_count", ascending=False).reset_index(drop=True)


def run_individual_id_for_species(
    species: str,
    experiment_dir: Path,
    models_root: Path = RESULTS_DATA_DIR / "sex_models_best",
    grid: pd.DataFrame | None = None,
    n_jobs: int = -1,
    random_state: int = 42,
) -> pd.DataFrame:
    """Run the individual identification experiment for a single species."""

    experiment_dir = Path(experiment_dir)
    experiment_dir.mkdir(parents=True, exist_ok=True)

    if grid is None:
        grid = _default_grid(random_state=random_state)

    species_dir = Path(SPLITS_DIR) / species
    df = (
        pd.read_parquet(species_dir / "train.parquet")
        .query("sex in ['f','m']")
        .drop(columns=["Fold"], errors="ignore")
    )

    if species == "eurasian_otter_(1)":
        feature_cols = [
            c
            for c in df.columns
            if c.startswith(("dist", "ang", "t")) and pd.api.types.is_numeric_dtype(df[c])
        ]
    else:
        feature_cols = [
            c for c in df.columns[5:] if pd.api.types.is_numeric_dtype(df[c])
        ]

    comps, _ = generate_pairwise_comparisons_from_df(
        df,
        id_col="individual_id",
        chunk_size=10,
        trail_size_list=[9, 7, 5, 3, 2],
        max_individuals=3,
        max_trails_per_animal=2,
        n_folds=3,
        random_state=random_state,
    )

    results_fp = experiment_dir / f"{species}_all_results_combined.parquet"
    if results_fp.exists():
        done = pd.read_parquet(results_fp, columns=grid.columns)
        done_keys = set(tuple(r) for r in done.values)
        todo = grid[grid.apply(lambda row: tuple(row.values) not in done_keys, axis=1)].reset_index(drop=True)
    else:
        todo = grid.copy()

    for _, row in todo.iterrows():
        fc = int(row.feature_count)
        sel = row.selection_method
        red = row.reducer
        nc = int(row.n_components)
        out = row.outlier_method
        scl = row.scaler_method
        sex = bool(row.use_sex)

        out_list = [out] if out is not None else [None]
        scl_list = [scl] if scl is not None else [None]
        desc = (
            f"FC={fc}|FS={sel}|Red={red}/{nc}|Out={out}|Scl={scl}|Sex={'on' if sex else 'off'}"
        )

        with tqdm_joblib(tqdm(desc=desc, total=len(comps), leave=False)):
            res = run_all_pairwise_projections_parallel(
                comparisons=comps,
                df=df,
                feature_cols=feature_cols,
                k_features=fc,
                selection_method=sel,
                reducers=[red],
                n_components=[nc],
                outlier_methods=out_list,
                scaler_methods=scl_list,
                use_sexmodel_prediction=sex,
                sexmodel_path=(models_root / f"{species}.joblib") if sex else None,
                debug=False,
                n_jobs=n_jobs,
            )

        df_run = pd.DataFrame(res)
        for col in grid.columns:
            df_run[col] = row[col]

        if results_fp.exists():
            df_full = pd.concat([pd.read_parquet(results_fp), df_run], ignore_index=True)
        else:
            df_full = df_run
        df_full.to_parquet(results_fp, index=False)

    return pd.read_parquet(results_fp)


def run_individual_id_all_species(
    experiment_dir: Path,
    models_root: Path = RESULTS_DATA_DIR / "sex_models_best",
    species_list: Iterable[str] | None = None,
    n_jobs: int = -1,
    random_state: int = 42,
) -> pd.DataFrame:
    """Run the experiment for all available species and combine the results."""

    experiment_dir = Path(experiment_dir)
    experiment_dir.mkdir(parents=True, exist_ok=True)

    if species_list is None:
        species_list = [p.name for p in Path(SPLITS_DIR).iterdir() if p.is_dir()]

    all_results = []
    grid = _default_grid(random_state=random_state)
    for sp in species_list:
        df_sp = run_individual_id_for_species(
            sp,
            experiment_dir=experiment_dir,
            models_root=models_root,
            grid=grid,
            n_jobs=n_jobs,
            random_state=random_state,
        )
        all_results.append(df_sp)

    df_all = pd.concat(all_results, ignore_index=True)
    df_all.to_parquet(experiment_dir / "all_species_results.parquet", index=False)
    return df_all

