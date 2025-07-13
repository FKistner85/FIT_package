"""Adapters for trail sampling and pairwise generation."""

from .trail_sampling import sample_trails
from .pair_generation import build_pairwise_comparisons
from .geometric_pairwise_projection import (
    generate_pairwise_comparisons_from_df as _generate_pw_from_df,
)
from typing import Dict, List, Tuple
import pandas as pd

__all__ = ["sample_trails", "build_pairwise_comparisons", "generate_pairwise_comparisons_from_df"]


def generate_pairwise_comparisons_from_df(
    df: pd.DataFrame,
    *,
    id_col: str = "individual_id",
    sample_col: str = "id",
    group_sizes: List[int] | None = None,
    n_repeats: int = 5,
    mode: str = "both",
    selfmatch_factor: float = 2.0,
    n_folds: int = 5,
    random_state: int = 0,
    show_progress: bool = False,
) -> Tuple[List[Dict], pd.DataFrame]:
    """Thin wrapper delegating to :func:`geometric_pairwise_projection.generate_pairwise_comparisons_from_df`."""
    comps = _generate_pw_from_df(
        df=df,
        id_col=id_col,
        sample_col=sample_col,
        group_sizes=group_sizes or [3, 5, 7, 10],
        n_repeats=n_repeats,
        mode=mode,
        selfmatch_factor=selfmatch_factor,
        n_folds=n_folds,
        random_state=random_state,
        show_progress=show_progress,
    )
    return comps, pd.DataFrame()
