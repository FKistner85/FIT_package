from itertools import combinations
from typing import Dict, List, Tuple, Union, Optional, Any, Iterable

import numpy as np
import pandas as pd
from FIT_python.config import GLOBAL_RANDOM_SEED
from FIT_python.soft_config import SOFT_CONFIG

__all__ = [
    "select_or_generate_trails",
    "generate_subsamples",
    "generate_pairs",
]


def _mode(values: Iterable[Any]) -> Any:
    """Return the first mode of ``values`` or ``pd.NA`` when empty."""
    ser = pd.Series(list(values)).dropna()
    if ser.empty:
        return pd.NA
    modes = ser.mode()
    return modes.iloc[0] if not modes.empty else pd.NA




def select_or_generate_trails(
    df: pd.DataFrame,
    *,
    strategy: str = "generate",
    individual_col: str = "individual_id",
    trail_col: str = "trail",
    sample_size: int = SOFT_CONFIG["pipeline_individual_id"]["trail_generation_defaults"]["sample_size"],
    random_state: int = GLOBAL_RANDOM_SEED,
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
        Seed for random sampling. Defaults to ``GLOBAL_RANDOM_SEED``.

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
    random_state: int = GLOBAL_RANDOM_SEED,
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
            df.loc[remaining, trail_col] = rng.choice(trails, size=len(remaining))

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


def generate_pairwise_comparisons_from_df(
    df: pd.DataFrame,
    *,
    trails: Optional[Dict[str, List[str]]] = None,
    strategy: str = "select",
    id_col: str = "individual_id",
    trail_col: str = "trail",
    fold_col: str = "fold",
    sex_col: str = "sex",
    id_field: str = "id",
    subsample: bool = False,
    evaluation: bool = False,
    random_state: int = GLOBAL_RANDOM_SEED,
) -> Tuple[List[Dict], pd.DataFrame]:
    """Create pairwise trail comparisons.

    The function uses :func:`select_or_generate_trails` and
    :func:`generate_subsamples` to obtain trail definitions. ``fold_a`` and
    ``fold_b`` are read directly from ``fold_col`` in ``df`` with
    ``same_fold`` indicating equality. If a ``substrate`` column exists or the
    species is ``"eurasian_otter"`` then ``substrate_a`` and ``substrate_b`` are
    added to each comparison.

    When ``evaluation`` is ``True`` the ``fold`` column is ignored and the
    returned comparisons contain ``None`` for ``fold_a``/``fold_b`` and do not
    set ``same_fold``.
    """

    df = select_or_generate_trails(
        df,
        strategy=strategy,
        individual_col=id_col,
        trail_col=trail_col,
        random_state=random_state,
    )

    if trails is None:
        if subsample:
            trails = generate_subsamples(
                df,
                trail_col=trail_col,
                id_col=id_field,
                individual_col=id_col,
                random_state=random_state,
            )
        else:
            trails = {
                tr: df.loc[df[trail_col] == tr, id_field].tolist()
                for tr in df[trail_col].dropna().unique()
            }

    pairs = generate_pairs(trails)

    base_to_ind = df.groupby(trail_col)[id_col].first().to_dict()
    if not evaluation and fold_col in df.columns:
        base_to_fold = df.groupby(trail_col)[fold_col].first().to_dict()
    else:
        base_to_fold = {}
    base_to_sex = df.groupby(trail_col)[sex_col].first().fillna("unknown").replace("", "unknown").to_dict()

    use_sub = "substrate" in df.columns or (
        "species" in df.columns and df["species"].eq("eurasian_otter").any()
    )
    base_to_sub = (
        df.groupby(trail_col)["substrate"].apply(_mode).to_dict() if use_sub and "substrate" in df.columns else {}
    )

    comparisons: List[Dict] = []
    for ta, tb in pairs:
        base_a = ta.split("_sub")[0]
        base_b = tb.split("_sub")[0]
        ind_a = base_to_ind.get(base_a, "unknown")
        ind_b = base_to_ind.get(base_b, "unknown")
        sex_a = base_to_sex.get(base_a, "unknown")
        sex_b = base_to_sex.get(base_b, "unknown")
        fold_a = base_to_fold.get(base_a)
        fold_b = base_to_fold.get(base_b)
        if evaluation:
            same_fold = None
            fold_val = None
        else:
            same_fold = fold_a == fold_b
            fold_val = (
                max(fold_a, fold_b)
                if fold_a is not None and fold_b is not None
                else fold_a or fold_b
            )
        rec = {
            "ind_a": ind_a,
            "ind_b": ind_b,
            "trail_a_id": ta,
            "trail_b_id": tb,
            "samples_a": trails[ta],
            "samples_b": trails[tb],
            "same_individual": (
                "unknown" if "unknown" in (ind_a, ind_b) else ind_a == ind_b
            ),
            "same_sex": (
                "unknown" if "unknown" in (sex_a, sex_b) else sex_a == sex_b
            ),
            "fold_a": fold_a,
            "fold_b": fold_b,
            "same_fold": same_fold,
            "fold": fold_val,
        }
        if use_sub:
            rec["substrate_a"] = base_to_sub.get(base_a)
            rec["substrate_b"] = base_to_sub.get(base_b)
        comparisons.append(rec)

    summary = pd.DataFrame({"n_pairs": [len(comparisons)], "n_trails": [len(trails)]})
    return comparisons, summary

