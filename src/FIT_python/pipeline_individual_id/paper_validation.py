"""Publication-oriented FIT-Py validation with train-only cutoff calibration.

This module intentionally does not modify legacy FIT outputs. In particular, the
legacy ``sequential_holdout.run`` may choose its Ward threshold using ``true_n``
of the evaluated holdout. This implementation NEVER uses outer holdout labels for
threshold selection. Any oracle estimate must be explicitly computed separately.

The canonical workflow is: known individuals -> outer grouped split -> inner
calibration pseudo-populations -> fixed cutoff -> untouched outer test.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.metrics import (
    balanced_accuracy_score, brier_score_loss, completeness_score,
    homogeneity_score, roc_auc_score, v_measure_score,
)
from sklearn.model_selection import StratifiedGroupKFold


@dataclass(frozen=True)
class PopulationCase:
    """A *training-only* inner calibration example with true trail identities.

    ``distances`` must have identical row/column trail IDs. ``identities``
    maps trail IDs to known animal IDs. Use pseudo-populations sampled only
    from the relevant outer training partition.
    """
    distances: pd.DataFrame
    identities: pd.Series
    case_id: str = "inner"


@dataclass(frozen=True)
class CalibrationResult:
    cutoff: float
    score: float
    grid: pd.DataFrame
    n_cases: int
    linkage_method: str
    missing_policy: str


def make_grouped_split_manifest(
    df: pd.DataFrame,
    *,
    individual_col: str = "individual_id",
    stratify_col: str = "sex",
    n_splits: int = 5,
    repeats: int = 3,
    seed: int = 42,
) -> pd.DataFrame:
    """Return deterministic *individual-disjoint* train/test manifests.

    The output contains one row per individual per outer split. Group labels
    may be inferred from print rows, but class labels must be consistent
    within each individual. All footprints of an individual remain together.
    """
    if individual_col not in df or stratify_col not in df:
        raise KeyError(f"Expected {individual_col!r} and {stratify_col!r}")
    sub = df[[individual_col, stratify_col]].dropna()
    if sub.empty:
        raise ValueError("No known individuals with stratification labels")
    cardinality = sub.groupby(individual_col)[stratify_col].nunique()
    if cardinality.gt(1).any():
        raise ValueError("Inconsistent stratification label within individual")
    animals = sub.drop_duplicates(individual_col).sort_values(individual_col)
    if animals[individual_col].nunique() < n_splits:
        raise ValueError("Fewer known individuals than requested folds")
    records: list[dict] = []
    X = animals[[individual_col]].to_numpy()
    y = animals[stratify_col].to_numpy()
    ids = animals[individual_col].astype(str).to_numpy()
    for rep in range(repeats):
        cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed + rep)
        for fold, (_, val_idx) in enumerate(cv.split(X, y, groups=ids)):
            test_ids = set(ids[val_idx])
            for ind in ids:
                records.append({"repeat": rep, "fold": fold, "individual_id": ind,
                                "partition": "test" if ind in test_ids else "train",
                                "seed": seed + rep})
    return pd.DataFrame.from_records(records)


def validate_symmetric_distances(distances: pd.DataFrame, *, tolerance: float = 1e-8) -> None:
    if not isinstance(distances, pd.DataFrame) or distances.shape[0] < 2:
        raise ValueError("Distances must be a square DataFrame of at least 2 trails")
    if distances.shape[0] != distances.shape[1]:
        raise ValueError("Distances must be square")
    if distances.index.has_duplicates or distances.columns.has_duplicates:
        raise ValueError("Trail names must be unique")
    if set(distances.index) != set(distances.columns):
        raise ValueError("Rows/columns have different trail identifiers")
    arr = distances.loc[distances.index, distances.index].to_numpy(dtype=float)
    nan_left, nan_right = np.isnan(arr), np.isnan(arr.T)
    if not np.array_equal(nan_left, nan_right) or not np.allclose(
        np.nan_to_num(arr), np.nan_to_num(arr.T), atol=tolerance, rtol=0
    ):
        raise ValueError("Distance matrix must be symmetric, including missing pairs")
    if not np.all(np.isfinite(arr) | np.isnan(arr)):
        raise ValueError("Infinite distance values are unsupported")
    if np.nanmin(arr) < -tolerance:
        raise ValueError("Negative distance")


def prepare_distance_matrix(
    distances: pd.DataFrame,
    *,
    missing_policy: str = "error",
    missing_constant: float | None = None,
) -> tuple[pd.DataFrame, float]:
    """Make a clustering matrix and report fraction of missing off-diagonal pairs.

    Missing pairs are a methodological choice; never silently replace them.
    ``max_observed`` is conservative but may force false splits, and should be
    included in a sensitivity analysis, not silently treated as ground truth.
    """
    validate_symmetric_distances(distances)
    ids = list(distances.index)
    a = distances.loc[ids, ids].to_numpy(dtype=float).copy()
    np.fill_diagonal(a, 0.0)
    upper = np.triu_indices(len(ids), k=1)
    missing = np.isnan(a[upper])
    missing_rate = float(missing.mean())
    if missing.any():
        if missing_policy == "error":
            raise ValueError("Missing pairwise distances: choose an explicit missing policy")
        observed = a[upper][~missing]
        if missing_policy == "max_observed":
            if observed.size == 0:
                raise ValueError("Cannot infer maximum: no observed pairwise distance")
            replacement = float(np.max(observed))
        elif missing_policy == "fixed":
            if missing_constant is None or not np.isfinite(missing_constant) or missing_constant < 0:
                raise ValueError("fixed missing policy requires non-negative missing_constant")
            replacement = float(missing_constant)
        else:
            raise ValueError(f"Unknown missing policy: {missing_policy}")
        a[np.isnan(a)] = replacement
    np.fill_diagonal(a, 0)
    return pd.DataFrame(a, index=ids, columns=ids), missing_rate


def cluster_with_cutoff(
    distances: pd.DataFrame,
    cutoff: float,
    *,
    method: str = "average",
    missing_policy: str = "error",
    missing_constant: float | None = None,
) -> tuple[pd.Series, float]:
    """Cluster without inspecting any true labels."""
    if not np.isfinite(cutoff) or cutoff <= 0:
        raise ValueError("Cutoff must be finite and positive")
    if method not in {"average", "complete", "single", "ward"}:
        raise ValueError("Unrecognised linkage method")
    d, missing_rate = prepare_distance_matrix(
        distances, missing_policy=missing_policy, missing_constant=missing_constant
    )
    # Ward assumes (squared) Euclidean geometry; distances derived from pairwise
    # LDA can violate this. Average linkage is the recommended sensitivity baseline.
    tree = linkage(squareform(d.to_numpy(), checks=False), method=method)
    clusters = fcluster(tree, t=cutoff, criterion="distance")
    return pd.Series(clusters, index=d.index, name="cluster"), missing_rate


def partition_metrics(predicted: pd.Series, truth: pd.Series) -> dict[str, float]:
    """Report both count error and grouping fidelity (no optimization here)."""
    common = predicted.index.intersection(truth.index)
    if len(common) < 2 or len(common) != len(predicted):
        raise ValueError("Ground truth must cover all clustered trails")
    p = predicted.loc[common].to_numpy()
    t = truth.loc[common].to_numpy()
    k_true = int(pd.Series(t).nunique())
    k_pred = int(pd.Series(p).nunique())
    if k_true <= 0:
        raise ValueError("No reference individuals")
    # False merging: different individuals grouped together.
    # False splitting: same individual assigned different clusters.
    merge_err = merge_n = split_err = split_n = 0
    for i, j in combinations(range(len(common)), 2):
        if t[i] == t[j]:
            split_n += 1
            split_err += p[i] != p[j]
        else:
            merge_n += 1
            merge_err += p[i] == p[j]
    return {
        "true_n": float(k_true), "pred_n": float(k_pred),
        "der": abs(k_pred - k_true) / k_true,
        "v_measure": float(v_measure_score(t, p)),
        "homogeneity": float(homogeneity_score(t, p)),
        "completeness": float(completeness_score(t, p)),
        "false_merge_rate": merge_err / merge_n if merge_n else float("nan"),
        "false_split_rate": split_err / split_n if split_n else float("nan"),
        "n_trails": float(len(common)),
    }


def _candidate_cutoffs(
    cases: Sequence[PopulationCase],
    *,
    method: str,
    missing_policy: str,
    missing_constant: float | None,
    max_candidates: int = 120,
) -> np.ndarray:
    heights = []
    for case in cases:
        d, _ = prepare_distance_matrix(case.distances, missing_policy=missing_policy,
                                       missing_constant=missing_constant)
        h = linkage(squareform(d.to_numpy(), checks=False), method=method)[:, 2]
        heights.extend(h[np.isfinite(h) & (h >= 0)].tolist())
    heights = np.unique(np.asarray(heights, dtype=float))
    if heights.size == 0:
        raise ValueError("No valid merge heights")
    # Evaluate midpoint of each interval, not an exact merge height (avoids
    # implementation-dependent threshold boundary behavior).
    mids = (heights[:-1] + heights[1:]) / 2 if len(heights) > 1 else np.array([])
    candidates = np.r_[max(1e-10, heights[0] * 0.5), mids, heights[-1] + max(1e-10, heights[-1] * 0.02)]
    candidates = np.unique(candidates[candidates > 0])
    if len(candidates) > max_candidates:
        candidates = np.unique(np.quantile(candidates, np.linspace(0, 1, max_candidates)))
    return candidates


def calibrate_cutoff_from_train_only(
    inner_training_cases: Sequence[PopulationCase],
    *,
    method: str = "average",
    missing_policy: str = "error",
    missing_constant: float | None = None,
    cutoffs: Iterable[float] | None = None,
    count_weight: float = 0.5,
) -> CalibrationResult:
    """Fit a global cutoff using *only* inner cases from the outer TRAIN set.

    The outer test matrix, test identities, and ``true_n`` must not be passed
    here. Precompute cases with inner reference groups that exclude the inner
    validation animals. Select by mean ``count_weight*(1-DER) +
    (1-count_weight)*V_measure`` across inner population trials.
    """
    cases = tuple(inner_training_cases)
    if not cases:
        raise ValueError("Provide at least one inner calibration case")
    if not 0 <= count_weight <= 1:
        raise ValueError("count_weight must be between 0 and 1")
    for case in cases:
        if len(case.identities) != len(case.distances) or not case.distances.index.isin(case.identities.index).all():
            raise ValueError(f"Incorrect identities in calibration case {case.case_id}")
    candidates = (np.asarray(list(cutoffs), dtype=float) if cutoffs is not None else
                  _candidate_cutoffs(cases, method=method, missing_policy=missing_policy,
                                     missing_constant=missing_constant))
    if not len(candidates) or not np.all(np.isfinite(candidates)) or np.any(candidates <= 0):
        raise ValueError("Invalid cutoff grid")
    rows = []
    for cutoff in candidates:
        scores = []
        for case in cases:
            cluster, _ = cluster_with_cutoff(case.distances, float(cutoff), method=method,
                                            missing_policy=missing_policy,
                                            missing_constant=missing_constant)
            m = partition_metrics(cluster, case.identities)
            scores.append(count_weight * (1-m["der"]) + (1-count_weight) * m["v_measure"])
        rows.append({"cutoff": float(cutoff), "mean_score": float(np.mean(scores)),
                     "sd_score": float(np.std(scores)), "n_cases": len(scores)})
    grid = pd.DataFrame(rows).sort_values("cutoff").reset_index(drop=True)
    best = grid[grid["mean_score"] >= grid["mean_score"].max() - 1e-12]
    # Choose median within an equally good plateau rather than smallest cutoff.
    chosen = float(best.iloc[len(best)//2]["cutoff"])
    return CalibrationResult(chosen, float(best["mean_score"].max()), grid, len(cases), method, missing_policy)


def evaluate_outer_population(
    outer_distances: pd.DataFrame,
    outer_truth: pd.Series,
    calibrated: CalibrationResult,
    *,
    missing_constant: float | None = None,
) -> dict[str, float]:
    """Use a FIXED inner-trained cutoff and inspect outer truth only afterwards."""
    clusters, missing_rate = cluster_with_cutoff(
        outer_distances, calibrated.cutoff, method=calibrated.linkage_method,
        missing_policy=calibrated.missing_policy, missing_constant=missing_constant
    )
    return {**partition_metrics(clusters, outer_truth),
            "cutoff": calibrated.cutoff, "missing_pair_fraction": missing_rate}


def pairwise_scores_to_distances(
    scores: pd.DataFrame, *, left: str = "trail_a_id", right: str = "trail_b_id",
    value: str = "dist_euclidean", trails: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Build symmetric matrix; absent comparisons stay NaN, not maximum."""
    required = {left, right, value}
    if not required.issubset(scores.columns):
        raise KeyError(f"Missing columns: {sorted(required.difference(scores.columns))}")
    names = sorted(set(scores[left].astype(str)) | set(scores[right].astype(str))) if trails is None else list(map(str, trails))
    if len(names) < 2:
        raise ValueError("At least two trails required")
    matrix = pd.DataFrame(np.nan, index=names, columns=names)
    np.fill_diagonal(matrix.values, 0)
    seen: dict[tuple[str, str], float] = {}
    for a, b, v in scores[[left, right, value]].itertuples(index=False, name=None):
        a, b = str(a), str(b)
        if a == b:
            continue
        if a not in matrix.index or b not in matrix.index:
            raise ValueError("Pair references trail outside declared names")
        if not pd.isna(v):
            pair = tuple(sorted((a,b)))
            if pair in seen and not np.isclose(float(v), seen[pair]):
                raise ValueError("Contradictory duplicate pair distances")
            seen[pair] = float(v)
            matrix.loc[a,b] = matrix.loc[b,a] = float(v)
    return matrix


def parse_same_individual(values: pd.Series) -> np.ndarray:
    """Parse repository labels, which are often serialized as 'True'/'False'."""
    labels = values.astype(str).str.strip().str.lower()
    valid = {"true": True, "1": True, "false": False, "0": False}
    if not labels.isin(valid).all():
        raise ValueError("same_individual must contain only True/False or 1/0")
    return labels.map(valid).to_numpy(dtype=bool)


def choose_binary_distance_cutoff(
    pairs: pd.DataFrame, *, distance_col: str = "dist_euclidean",
    truth_col: str = "same_individual",
) -> float:
    """Tune same/different threshold on training pairs ONLY."""
    x = pairs[[distance_col, truth_col]].dropna()
    values = x[distance_col].to_numpy(dtype=float)
    truth = parse_same_individual(x[truth_col])
    if len(values) < 2 or len(np.unique(truth)) < 2:
        raise ValueError("Training pairs must include both same and different examples")
    unique = np.unique(values)
    cuts = np.unique(np.r_[unique[0] - 1e-9,
        (unique[1:] + unique[:-1])/2, unique[-1] + 1e-9])
    performances = [balanced_accuracy_score(truth, values <= c) for c in cuts]
    best = np.flatnonzero(np.isclose(performances, max(performances)))
    return float(cuts[best[len(best)//2]])


def evaluate_pair_classifier(
    outer_pairs: pd.DataFrame, threshold: float, *,
    distance_col: str = "dist_euclidean", truth_col: str = "same_individual",
) -> dict[str, float]:
    x = outer_pairs[[distance_col, truth_col]].dropna()
    if x.empty:
        raise ValueError("No computable test pairs")
    vals = x[distance_col].to_numpy(dtype=float)
    truth = parse_same_individual(x[truth_col])
    pred = vals <= threshold
    return {"balanced_accuracy": float(balanced_accuracy_score(truth, pred)),
            "false_merge_rate": float(np.mean(pred[~truth])) if (~truth).any() else float("nan"),
            "false_split_rate": float(np.mean(~pred[truth])) if truth.any() else float("nan"),
            "auc": float(roc_auc_score(truth, -vals)) if np.unique(truth).size == 2 else float("nan"),
            "n_pairs": float(len(x)), "pair_coverage": float(len(x) / len(outer_pairs))}


def make_pseudo_trails(
    df: pd.DataFrame, *,
    print_id_col: str = "id", individual_col: str = "individual_id",
    trail_col: str = "trail", prints_per_trail: int = 4,
    trails_per_individual: int = 2, random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate **disjoint** artificial trails from known individuals.

    Animals with fewer than ``prints_per_trail * trails_per_individual``
    footprints are ineligible and counted in the returned eligibility report.
    Original trails may be mixed; the output is an artificial benchmark and
    must never be presented as actual field trails.
    """
    if prints_per_trail < 1 or trails_per_individual < 2:
        raise ValueError("Need at least one print and two separate trails per individual")
    if df[print_id_col].duplicated().any():
        raise ValueError("Print ID must be unique")
    rng = np.random.default_rng(random_state)
    samples, report = [], []
    for ind, grp in df.groupby(individual_col, sort=True):
        n_need = prints_per_trail * trails_per_individual
        eligible = len(grp) >= n_need
        report.append({"individual_id": str(ind), "available_prints": len(grp),
                       "needed_prints": n_need, "eligible": eligible})
        if not eligible:
            continue
        chosen = rng.choice(grp.index.to_numpy(), size=n_need, replace=False)
        for j in range(trails_per_individual):
            sub = df.loc[chosen[j*prints_per_trail:(j+1)*prints_per_trail]].copy()
            sub[trail_col] = f"paper_{str(ind)}_rep{j}"
            samples.append(sub)
    if not samples:
        raise ValueError("No eligible individuals at requested trail length")
    result = pd.concat(samples, ignore_index=True)
    if result[print_id_col].duplicated().any():
        raise AssertionError("A print occurred in more than one synthetic trail")
    return result, pd.DataFrame(report)


def apply_missingness(
    df: pd.DataFrame, *, features: Sequence[str], fraction: float,
    strategy: str = "mcar", feature_groups: Mapping[str, Sequence[str]] | None = None,
    random_state: int = 42,
) -> pd.DataFrame:
    """Artificial missingness stress test; does NOT simulate field pixels.

    ``block`` removes named feature groups, optionally representing the
    measurements affected by an anatomical landmark. Landmark-to-measurement
    mappings must be supplied explicitly; no biological mapping is invented.
    Must impute via a fitted *training-only* imputer in the calling experiment.
    """
    if not 0 <= fraction <= 1:
        raise ValueError("fraction must be between 0 and 1")
    rng = np.random.default_rng(random_state)
    out = df.copy()
    if strategy == "mcar":
        mask = rng.random((len(out), len(features))) < fraction
        out.loc[:, list(features)] = out[list(features)].mask(mask)
    elif strategy == "block":
        if not feature_groups:
            raise ValueError("block missingness needs explicitly supplied feature_groups")
        groups = [list(cols) for cols in feature_groups.values()]
        if any(not set(cols).issubset(features) for cols in groups):
            raise ValueError("feature groups contain unknown features")
        for idx in out.index:
            if rng.random() < fraction:
                cols = groups[int(rng.integers(len(groups)))]
                out.loc[idx, cols] = np.nan
    else:
        raise ValueError("Unknown missingness strategy")
    return out


def impute_from_train_only(
    train: pd.DataFrame, test: pd.DataFrame, features: Sequence[str], *,
    strategy: str = "median", seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit numeric imputer only on known training footprints."""
    if strategy == "median":
        from sklearn.impute import SimpleImputer
        model = SimpleImputer(strategy="median", keep_empty_features=True)
    elif strategy == "forest":
        from sklearn.experimental import enable_iterative_imputer  # noqa: F401
        from sklearn.impute import IterativeImputer
        from sklearn.ensemble import RandomForestRegressor
        model = IterativeImputer(
            estimator=RandomForestRegressor(n_estimators=40, random_state=seed,
                                             n_jobs=1),
            max_iter=5, random_state=seed, skip_complete=True,
        )
    else:
        raise ValueError("Use 'median' or 'forest'")
    train_copy, test_copy = train.copy(), test.copy()
    model.fit(train[list(features)])
    train_copy.loc[:, list(features)] = model.transform(train[list(features)])
    test_copy.loc[:, list(features)] = model.transform(test[list(features)])
    return train_copy, test_copy


def evaluate_sex_oof_calibration(
    predictions: pd.DataFrame, *, individual_col: str = "individual_id",
    truth_col: str = "sex", male_probability_col: str = "p_male",
    is_out_of_fold_col: str = "is_out_of_fold", n_bins: int = 8,
) -> tuple[dict[str, float], pd.DataFrame]:
    """Compute calibration on individual-aggregated **out-of-fold** predictions.

    Refuse in-sample predictions, as they yield misleadingly good reliability.
    """
    for c in [individual_col, truth_col, male_probability_col, is_out_of_fold_col]:
        if c not in predictions:
            raise KeyError(f"Missing {c!r}")
    if not predictions[is_out_of_fold_col].astype(bool).all():
        raise ValueError("Only out-of-fold predictions may be used")
    sex = predictions[truth_col].astype(str).str.lower()
    known = {"male":1, "m":1, "female":0, "f":0, "1":1, "0":0}
    if not sex.isin(known).all():
        raise ValueError("Sex labels must be male/female or M/F")
    tmp = predictions[[individual_col, male_probability_col]].copy()
    tmp["sex_numeric"] = sex.map(known).astype(int)
    if not tmp[male_probability_col].between(0,1).all():
        raise ValueError("Probabilities outside [0,1]")
    if tmp.groupby(individual_col)["sex_numeric"].nunique().gt(1).any():
        raise ValueError("Conflicting sex labels for one individual")
    grouped = tmp.groupby(individual_col).agg(y=("sex_numeric", "first"),
                                               p=(male_probability_col,"mean"))
    brier = float(brier_score_loss(grouped["y"], grouped["p"]))
    bins = np.linspace(0, 1, n_bins+1)
    groups = pd.cut(grouped["p"], bins=bins, include_lowest=True, labels=False)
    stats = grouped.assign(bin=groups).groupby("bin", observed=True).agg(
        n=("y","size"), mean_probability=("p","mean"), observed_frequency=("y","mean")
    ).reset_index()
    ece = float((stats["n"] * (stats["mean_probability"]-stats["observed_frequency"]).abs()).sum() / len(grouped))
    return {"brier": brier, "ece": ece, "n_individuals": float(len(grouped))}, stats
