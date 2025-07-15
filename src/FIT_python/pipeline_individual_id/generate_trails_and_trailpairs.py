from itertools import combinations
from typing import Dict, List, Tuple, Union

import numpy as np
import pandas as pd
from FIT_python.soft_config import SOFT_CONFIG

__all__ = [
    "select_or_generate_trails",
    "generate_subsamples",
    "generate_pairs",
]



def select_or_generate_trails(
    df: pd.DataFrame,
    *,
    strategy: str = "generate",
    individual_col: str = "individual_id",
    trail_col: str = "trail",
    sample_size: int = SOFT_CONFIG["pipeline_individual_id"]["trail_generation_defaults"]["sample_size"],
    random_state: int = 0,
) -> pd.DataFrame:
    """Return existing trails or generate new ones.

    Parameters
    ----------
    df : pd.DataFrame
        Input data with one row per footprint.
    strategy : str, optional
        ``"select"`` to keep existing ``trail`` values or ``"generate``" to
        create new trails. Defaults to ``"generate"``.
    individual_col : str, optional
        Column identifying individuals. Defaults to ``"individual_id"``.
    trail_col : str, optional
        Column containing trail identifiers. Defaults to ``"trail"``.
    sample_size : int, optional
        Number of observations per generated trail. Defaults to
        ``SOFT_CONFIG['pipeline_individual_id']['trail_generation_defaults']['sample_size']``.
    random_state : int, optional
        Seed for random sampling. Defaults to ``0``.

    Returns
    -------
    pd.DataFrame
        ``df`` with updated ``trail_col`` values.
    """

    df = df.copy()
    rng = np.random.default_rng(random_state)

    if strategy == "select" and trail_col in df.columns:
        return df

    df[trail_col] = pd.NA
    for ind, grp in df.groupby(individual_col):
        idxs = grp.index.tolist()
        rng.shuffle(idxs)
        n_trails = len(idxs) // sample_size
        for i in range(n_trails):
            sel = idxs[i * sample_size : (i + 1) * sample_size]
            trail_id = f"{ind}_{sample_size}_{i + 1}"
            df.loc[sel, trail_col] = trail_id

    return df


def generate_subsamples(
    df: pd.DataFrame,
    *,
    trail_col: str = "trail",
    id_col: str = "id",
    individual_col: str = "individual_id",
    sample_size: int = SOFT_CONFIG["pipeline_individual_id"]["trail_generation_defaults"]["sample_size"],
    subsample_sizes: Tuple[int, ...] = tuple(SOFT_CONFIG["pipeline_individual_id"]["trail_generation_defaults"]["subsample_sizes"]),
    n_candidates: int = SOFT_CONFIG["pipeline_individual_id"]["trail_generation_defaults"]["n_candidates"],
    random_state: int = 0,
) -> Dict[str, List[str]]:
    """Create diverse subsamples for each trail."""

    df = df.copy()
    rng = np.random.default_rng(random_state)

    for ind, grp in df.groupby(individual_col):
        trails = grp[trail_col].dropna().unique().tolist()
        if not trails:
            continue
        remaining = grp[grp[trail_col].isna()].index.tolist()
        if remaining:
            assign = rng.choice(trails, size=len(remaining))
            for idx, tr in zip(remaining, assign):
                df.at[idx, trail_col] = tr

    trail_to_ids = {
        tr: df.loc[df[trail_col] == tr, id_col].tolist()
        for tr in df[trail_col].dropna().unique()
    }

    subsamples: Dict[str, List[str]] = {}
    for tr, ids in trail_to_ids.items():
        for ss in subsample_sizes:
            if len(ids) < ss:
                continue
            cand = [list(rng.choice(ids, size=ss, replace=False)) for _ in range(n_candidates)]
            sets = [set(c) for c in cand]
            sims = np.zeros(len(cand))
            for i, a in enumerate(sets):
                if len(cand) == 1:
                    sims[i] = 0.0
                else:
                    acc = 0.0
                    for j, b in enumerate(sets):
                        if i == j:
                            continue
                        inter = len(a & b)
                        union = len(a | b)
                        acc += inter / union if union else 1.0
                    sims[i] = acc / (len(cand) - 1)
            order = np.argsort(sims)
            chosen = []
            for idx in order:
                c = cand[idx]
                if any(set(c) == set(o) for o in chosen):
                    continue
                chosen.append(c)
                if len(chosen) == 3:
                    break
            for i, sub in enumerate(chosen, 1):
                name = f"{tr}_sub{ss}_sample{i}"
                subsamples[name] = sub

    return subsamples


def generate_pairs(trails: Dict[str, List[str]]) -> List[Tuple[str, str]]:
    """Create all trail pair combinations excluding same base trail."""

    names = list(trails.keys())
    pairs: List[Tuple[str, str]] = []
    for i in range(len(names)):
        base_i = names[i].split("_sub")[0]
        for j in range(i + 1, len(names)):
            base_j = names[j].split("_sub")[0]
            if base_i == base_j:
                continue
            pairs.append((names[i], names[j]))
    return pairs




