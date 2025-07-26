"""Configuration helper for individual ID hyperparameter search."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd
from sklearn.model_selection import PredefinedSplit
from sklearn.pipeline import Pipeline
from skopt import BayesSearchCV
from skopt.space import Categorical

from FIT_python.general_pipeline_steps.outlier_wrapper import OutlierCleanerTransformer
from FIT_python.general_pipeline_steps.feature_scaler_wrapper import FeatureScalerTransformer
from FIT_python.general_pipeline_steps.feature_selection_wrapper import FeatureSelectionTransformer
from FIT_python.general_pipeline_steps.dimensionality_reduction_wrapper import DimensionalityReducerTransformer
from FIT_python.pipeline_individual_id.pairwise_individual_id_pipeline import (
    run_all_pairwise_projections_parallel,
)
from FIT_python.pipeline_individual_id.generate_trails_and_trailpairs import (
    generate_pairwise_comparisons_from_df,
)
from FIT_python.pipeline_individual_id.evaluation import separation_score
from FIT_python.soft_config import SOFT_CONFIG
from FIT_python.config import SPLITS_DIR, RESULTS_DATA_DIR, GLOBAL_RANDOM_SEED
from FIT_python.data_split_and_summary.data_import_utils import get_feature_cols


PIPE_CFG = SOFT_CONFIG["pipeline_individual_id"]

SEARCH_SPACE_CFG = PIPE_CFG.get("search_spaces", {})

SEARCH_SPACES = {
    "est__outlier_method": Categorical(SEARCH_SPACE_CFG.get("outlier", [None])),
    "est__scaler_method": Categorical(SEARCH_SPACE_CFG.get("scale", [None])),
    "est__selection_method": Categorical(
        SEARCH_SPACE_CFG.get("select__method", [None])
    ),
    "est__k_features": Categorical(SEARCH_SPACE_CFG.get("select__k", [5])),
    "est__reducer": Categorical(SEARCH_SPACE_CFG.get("reduce__method", ["pca"])),
    "est__n_components": Categorical(SEARCH_SPACE_CFG.get("n_components", [2])),
    "est__use_sexmodel_prediction": Categorical(
        SEARCH_SPACE_CFG.get("use_sexmodel_prediction", [False, True])
    ),
}


class PairwiseEstimator:
    """Estimator that computes pairwise distances using the embedding pipeline."""

    def __init__(
        self,
        feature_cols: Iterable[str],
        *,
        outlier_method: str | None = None,
        scaler_method: str | None = None,
        selection_method: str | None = None,
        k_features: int = 5,
        reducer: str = "pca",
        n_components: int = 2,
        use_sexmodel_prediction: bool = False,
        sexmodel_path: str | None = None,
    ) -> None:
        self.feature_cols = list(feature_cols)
        self.outlier_method = outlier_method
        self.scaler_method = scaler_method
        self.selection_method = selection_method
        self.k_features = k_features
        self.reducer = reducer
        self.n_components = n_components
        self.use_sexmodel_prediction = use_sexmodel_prediction
        self.sexmodel_path = sexmodel_path

        self._train_df: pd.DataFrame | None = None
        self.results_: pd.DataFrame | None = None

    def get_params(self, deep: bool = True):  # pragma: no cover - simple passthrough
        return {
            "feature_cols": self.feature_cols,
            "outlier_method": self.outlier_method,
            "scaler_method": self.scaler_method,
            "selection_method": self.selection_method,
            "k_features": self.k_features,
            "reducer": self.reducer,
            "n_components": self.n_components,
            "use_sexmodel_prediction": self.use_sexmodel_prediction,
            "sexmodel_path": self.sexmodel_path,
        }

    def set_params(self, **params):  # pragma: no cover - simple passthrough
        for key, val in params.items():
            setattr(self, key, val)
        return self

    def fit(self, X: pd.DataFrame, y=None):
        self._train_df = pd.DataFrame(X).copy()
        return self

    def predict(self, X: pd.DataFrame):
        if self._train_df is None:
            raise RuntimeError("Estimator has not been fitted")

        df_val = pd.DataFrame(X).copy()
        df_all = pd.concat([self._train_df, df_val], ignore_index=True)
        comps, _ = generate_pairwise_comparisons_from_df(df_val)
        res = run_all_pairwise_projections_parallel(
            comps,
            df_all,
            feature_cols=self.feature_cols,
            k_features=self.k_features,
            reducers=[self.reducer],
            selection_method=self.selection_method,
            n_components=self.n_components,
            outlier_methods=self.outlier_method,
            scaler_methods=self.scaler_method,
            use_sexmodel_prediction=self.use_sexmodel_prediction,
            sexmodel_path=self.sexmodel_path,
            n_jobs=1,
        )
        self.results_ = pd.DataFrame(res)
        return self.results_


def run_species_search(
    *,
    species_filter: list[str] | None = None,
    n_iter: int = 2,
    cv: int | str = "fold",
    random_state: int = GLOBAL_RANDOM_SEED,
) -> None:
    """Run BayesSearchCV for all species in ``SPLITS_DIR``.

    Parameters
    ----------
    species_filter : list[str], optional
        If given, restrict the search to these species directory names.
    n_iter : int, optional
        Number of parameter samples drawn by :class:`skopt.BayesSearchCV`.
    cv : int or str, optional
        Cross-validation strategy. ``"fold"`` uses the ``Fold`` column via
        :class:`~sklearn.model_selection.PredefinedSplit`. Any other value is
        forwarded to :class:`skopt.BayesSearchCV` as-is.
    random_state : int, optional
        Random seed controlling the search. Defaults to ``GLOBAL_RANDOM_SEED``.
    """

    for sp_dir in sorted(SPLITS_DIR.iterdir()):
        if not sp_dir.is_dir():
            continue
        species = sp_dir.name
        if species_filter and species not in species_filter:
            continue

        train_fp = sp_dir / "train.parquet"
        if not train_fp.exists():
            continue

        df_train = pd.read_parquet(train_fp)
        if cv == "fold":
            fold_ids = df_train["Fold"].astype(int).to_numpy()
            df_train = df_train.drop(columns=["Fold"])
            cv_strategy = PredefinedSplit(test_fold=fold_ids)
        else:
            df_train = df_train.drop(columns=["Fold"], errors="ignore")
            cv_strategy = cv
        feature_cols = get_feature_cols(df_train)

        pipe = Pipeline([
            ("outlier", OutlierCleanerTransformer()),
            ("scale", FeatureScalerTransformer()),
            ("select", FeatureSelectionTransformer()),
            ("reduce", DimensionalityReducerTransformer()),
            ("est", PairwiseEstimator(feature_cols)),
        ])

        search = BayesSearchCV(
            estimator=pipe,
            search_spaces=SEARCH_SPACES,
            n_iter=n_iter,
            scoring=separation_score(),
            refit=False,
            cv=cv_strategy,
            n_jobs=1,
            random_state=random_state,
            verbose=0,
        )

        extra_cols = ["individual_id", "Trail", "id"]
        X_all = df_train[feature_cols + extra_cols].copy()
        search.fit(X_all, None)

        out_dir = RESULTS_DATA_DIR / f"{species}_id_search"
        out_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(search.cv_results_).to_csv(out_dir / "cv_results.csv", index=False)

        print(f"✅ Finished {species} -> results under {out_dir}")


