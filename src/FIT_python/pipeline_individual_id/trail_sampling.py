from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

__all__ = ["sample_trails"]


def sample_trails(
    df: pd.DataFrame,
    group_col: str,
    window_lengths: List[int],
    n_windows: int,
    random_state: int,
    *,
    id_col: str = "id",
) -> Dict[str, Dict[int, List[Tuple[int, List[str]]]]]:
    """Randomly sample trails for each individual.

    The largest ``window_length`` defines the base trail size. All
    observations of an individual are shuffled and split into as many full
    trails of this size as possible **without replacement**. Remaining
    samples are distributed evenly across the trails.  Sub-trails of the
    requested lengths are then drawn at random from these base trails.
    """

    rng = np.random.default_rng(random_state)

    max_len = max(window_lengths)
    result: Dict[str, Dict[int, List[Tuple[int, List[str]]]]] = {}
    for gid, grp in df.groupby(group_col):
        idxs = grp[id_col].astype(str).tolist()
        rng.shuffle(idxs)
        gid = str(gid)

        n_base = len(idxs) // max_len
        if n_base == 0:
            result[gid] = {L: [] for L in window_lengths}
            continue

        base_trails = [idxs[i * max_len : (i + 1) * max_len] for i in range(n_base)]
        leftover = idxs[n_base * max_len :]
        for i, obs in enumerate(leftover):
            base_trails[i % n_base].append(obs)

        per_size: Dict[int, List[Tuple[int, List[str]]]] = {
            L: [] for L in window_lengths
        }
        for bi, base in enumerate(base_trails):
            per_size[max_len].append((bi, base[:max_len]))
            for L in window_lengths:
                if L == max_len or len(base) < L:
                    continue
                for rep in range(n_windows):
                    subset = list(rng.choice(base, size=L, replace=True))
                    per_size[L].append((bi * n_windows + rep, subset))

        result[gid] = per_size

    return result
