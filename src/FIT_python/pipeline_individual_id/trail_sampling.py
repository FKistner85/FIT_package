from itertools import combinations
from typing import Dict, List
import math

import numpy as np
import pandas as pd

__all__ = ["sample_trails"]


def sample_trails(
    df: pd.DataFrame,
    group_col: str,
    window_lengths: List[int],
    N_pool: int,
    n_windows: int,
    random_state: int,
    *,
    id_col: str = "id",
) -> Dict[str, Dict[int, List[List[str]]]]:
    """Sample diverse ``id`` subsets for each individual.

    The function draws a pool of candidate windows for every ``group_col`` and
    ``window_length``. The average Jaccard dissimilarity to all other
    windows in the pool is computed and the ``n_windows`` most diverse
    subsets are returned. Windows are returned as lists of ``id`` values so
    the original row order can be restored later.
    """

    rng = np.random.default_rng(random_state)

    pools: Dict[str, Dict[int, List[List[str]]]] = {}
    for gid, grp in df.groupby(group_col):
        idxs = sorted(grp[id_col].astype(str).tolist())
        gid = str(gid)
        pools[gid] = {}
        for L in window_lengths:
            if len(idxs) < L:
                pools[gid][L] = []
                continue

            max_pool = math.comb(len(idxs), L)
            if max_pool <= N_pool:
                pool_windows = [list(c) for c in combinations(idxs, L)]
            else:
                seen = set()
                pool_windows = []
                while len(pool_windows) < N_pool:
                    cand = tuple(sorted(rng.choice(idxs, size=L, replace=False)))
                    if cand in seen:
                        continue
                    seen.add(cand)
                    pool_windows.append(list(cand))

            sets = [set(w) for w in pool_windows]
            n = len(sets)
            if n == 1:
                diversity = np.array([0.0])
            else:
                diversity = np.zeros(n)
                for i in range(n):
                    acc = 0.0
                    for j in range(n):
                        if i == j:
                            continue
                        inter = len(sets[i] & sets[j])
                        union = len(sets[i] | sets[j])
                        sim = inter / union if union else 1.0
                        acc += 1.0 - sim
                    diversity[i] = acc / (n - 1)

            select_idx = np.argsort(-diversity)[:n_windows]
            pools[gid][L] = [pool_windows[i] for i in select_idx]

    return pools
