from __future__ import annotations

from typing import Callable, Iterable, Sequence

import numpy as np
import pandas as pd


def _validate_probability_array(proba: Sequence[float]) -> np.ndarray:
    """Return ``proba`` as a float array and validate the range.

    Parameters
    ----------
    proba:
        Iterable of probabilities that are expected to lie within ``[0, 1]``.

    Returns
    -------
    numpy.ndarray
        One-dimensional float array.
    """

    arr = np.asarray(proba, dtype=float)
    if arr.ndim != 1:
        raise ValueError("Probability array must be one-dimensional")
    if not np.isfinite(arr).all():
        raise ValueError("Probability array contains NaN or infinite values")
    if (arr < 0).any() or (arr > 1).any():
        raise ValueError("Probabilities must be within [0, 1]")
    return arr


def _validate_binary_array(values: Sequence[float]) -> np.ndarray:
    """Return ``values`` as a binary integer array (0/1)."""

    arr = np.asarray(values)
    if arr.ndim != 1:
        raise ValueError("Binary targets must be one-dimensional")

    if arr.dtype == bool:
        return arr.astype(int)

    if np.issubdtype(arr.dtype, np.integer):
        unique = np.unique(arr)
        if not set(unique).issubset({0, 1}):
            raise ValueError("Binary targets must only contain 0/1 values")
        return arr.astype(int)

    if np.issubdtype(arr.dtype, np.floating):
        rounded = np.round(arr).astype(int)
        if not np.allclose(arr, rounded, atol=1e-8):
            raise ValueError("Binary targets must only contain 0/1 values")
        if not set(np.unique(rounded)).issubset({0, 1}):
            raise ValueError("Binary targets must only contain 0/1 values")
        return rounded

    raise ValueError(
        "Binary targets must be bool, integer or float values containing only 0/1"
    )


def build_calibration_table(
    y_true: Sequence[float],
    proba: Sequence[float],
    *,
    n_bins: int = 10,
    strategy: str = "quantile",
) -> pd.DataFrame:
    """Return a calibration table with bin statistics.

    The returned dataframe contains the lower/upper bin edges, the number of
    samples per bin, the mean predicted probability and the empirical fraction
    of positives.
    """

    if n_bins <= 0:
        raise ValueError("n_bins must be a positive integer")
    if strategy not in {"quantile", "uniform"}:
        raise ValueError("strategy must be 'quantile' or 'uniform'")

    y = _validate_binary_array(y_true)
    p = _validate_probability_array(proba)
    if len(y) != len(p):
        raise ValueError("y_true and proba must contain the same number of samples")
    if len(y) == 0:
        return pd.DataFrame(
            columns=[
                "bin",
                "bin_lower",
                "bin_upper",
                "count",
                "mean_predicted",
                "fraction_of_positives",
            ]
        )

    df = pd.DataFrame({"target": y, "proba": p})

    if strategy == "quantile":
        q = min(n_bins, df["proba"].nunique())
        if q <= 1:
            q = 1
        df["bin"] = pd.qcut(df["proba"], q=q, duplicates="drop")
    else:
        edges = np.linspace(0.0, 1.0, n_bins + 1)
        df["bin"] = pd.cut(df["proba"], bins=edges, include_lowest=True)

    if df["bin"].isna().all():
        raise ValueError("Could not create calibration bins – check probability input")

    bin_codes = df["bin"].cat.codes
    grouped = df.groupby(bin_codes, observed=True)

    table = grouped.agg(
        count=("target", "size"),
        mean_predicted=("proba", "mean"),
        fraction_of_positives=("target", "mean"),
    )
    table.reset_index(drop=True, inplace=True)

    categories = df["bin"].cat.categories
    lowers = []
    uppers = []
    for interval in categories:
        lowers.append(float(interval.left))
        uppers.append(float(interval.right))

    table.insert(0, "bin", np.arange(len(table)))
    table.insert(1, "bin_lower", lowers[: len(table)])
    table.insert(2, "bin_upper", uppers[: len(table)])

    return table


def expected_calibration_error(
    y_true: Sequence[float],
    proba: Sequence[float],
    *,
    n_bins: int = 10,
    strategy: str = "quantile",
) -> float:
    """Return the Expected Calibration Error (ECE)."""

    table = build_calibration_table(y_true, proba, n_bins=n_bins, strategy=strategy)
    if table.empty:
        return float("nan")
    total = table["count"].sum()
    weights = table["count"] / total
    diff = np.abs(table["fraction_of_positives"] - table["mean_predicted"])
    return float(np.sum(diff * weights))


def bootstrap_confidence_interval(
    metric_fn: Callable[..., float],
    *arrays: Iterable[float],
    n_bootstraps: int = 1000,
    confidence: float = 0.95,
    random_state: int | None = None,
) -> tuple[float, float]:
    """Return a bootstrap confidence interval for ``metric_fn``.

    Parameters
    ----------
    metric_fn:
        Callable applied to each bootstrap sample. It receives the sampled
        arrays in the same order as provided via ``arrays``.
    arrays:
        Arrays of equal length from which bootstrap samples are drawn.
    n_bootstraps:
        Number of bootstrap iterations.
    confidence:
        Confidence level between ``0`` and ``1``. Defaults to ``0.95``.
    random_state:
        Optional seed used for the bootstrap sampling.
    """

    if n_bootstraps <= 0:
        raise ValueError("n_bootstraps must be greater than zero")
    if not 0 < confidence < 1:
        raise ValueError("confidence must lie within (0, 1)")
    if not arrays:
        raise ValueError("At least one data array is required")

    samples = [np.asarray(a) for a in arrays]
    lengths = {len(a) for a in samples}
    if len(lengths) != 1:
        raise ValueError("All arrays must share the same length")
    n_samples = lengths.pop()
    if n_samples == 0:
        return float("nan"), float("nan")

    rng = np.random.default_rng(random_state)
    stats = np.empty(n_bootstraps, dtype=float)
    for i in range(n_bootstraps):
        idx = rng.integers(0, n_samples, size=n_samples)
        boot_data = [a[idx] for a in samples]
        stats[i] = metric_fn(*boot_data)

    alpha = (1 - confidence) / 2
    lower = float(np.quantile(stats, alpha))
    upper = float(np.quantile(stats, 1 - alpha))
    return lower, upper


def assign_confidence_bands(
    max_proba: Sequence[float],
    *,
    boundaries: Sequence[float] = (0.6, 0.8, 0.9),
    labels: Sequence[str] | None = None,
) -> pd.Categorical:
    """Return discrete confidence bands for ``max_proba``.

    Parameters
    ----------
    max_proba:
        Maximum class probability per sample.
    boundaries:
        Sorted sequence of boundary values in ``(0, 1)`` defining the
        confidence bins. Values lower than the first boundary fall into the
        first bin, values above the last boundary form the final bin.
    labels:
        Optional custom labels for the bins. When omitted a human-readable
        representation of the bin ranges is generated automatically.
    """

    arr = _validate_probability_array(max_proba)
    if arr.size == 0:
        return pd.Categorical([], categories=[], ordered=True)

    if any(b <= 0 or b >= 1 for b in boundaries):
        raise ValueError("boundaries must lie strictly within (0, 1)")

    # Ensure increasing order and uniqueness
    boundaries = tuple(sorted(set(float(b) for b in boundaries)))
    if any(b1 >= b2 for b1, b2 in zip(boundaries, boundaries[1:])):
        raise ValueError("boundaries must be strictly increasing")

    edges = np.array((0.0, *boundaries, 1.0), dtype=float)
    if np.any(np.diff(edges) <= 0):
        raise ValueError("boundaries must define increasing bins")

    if labels is None:
        labels = []
        for lower, upper in zip(edges[:-1], edges[1:]):
            if lower == 0.0:
                labels.append(f"<{upper:.2f}")
            elif upper == 1.0:
                labels.append(f">={lower:.2f}")
            else:
                labels.append(f"[{lower:.2f}, {upper:.2f})")
    else:
        if len(labels) != len(edges) - 1:
            raise ValueError("labels must match the number of bins")

    cat = pd.cut(
        arr,
        bins=edges,
        labels=list(labels),
        include_lowest=True,
        right=False,
        ordered=True,
    )
    return cat
