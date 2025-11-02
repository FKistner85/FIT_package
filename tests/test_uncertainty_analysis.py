import numpy as np
import pandas as pd
import pytest

from FIT_python.analysis.uncertainty import (
    assign_confidence_bands,
    bootstrap_confidence_interval,
    build_calibration_table,
    expected_calibration_error,
)
from FIT_python.pipeline_sex.sex_predict_and_visualisation import summarise_uncertainty
from FIT_python.pipeline_individual_id.uncertainty import summarise_pairwise_uncertainty


def test_build_calibration_table_quantile_strategy():
    y_true = [0, 1, 0, 1]
    proba = [0.1, 0.9, 0.2, 0.8]
    table = build_calibration_table(y_true, proba, n_bins=2, strategy="quantile")
    assert table["count"].sum() == len(y_true)
    assert table["bin"].tolist() == [0, 1]


def test_expected_calibration_error_zero_when_perfect():
    y_true = [0, 1, 0, 1]
    proba = [0.0, 1.0, 0.0, 1.0]
    ece = expected_calibration_error(y_true, proba, n_bins=2, strategy="uniform")
    assert ece == pytest.approx(0.0)


def test_assign_confidence_bands_returns_ordered_categories():
    bands = assign_confidence_bands([0.55, 0.75, 0.95], boundaries=(0.6, 0.8))
    assert list(bands.categories) == ["<0.60", "[0.60, 0.80)", ">=0.80"]
    assert list(bands) == ["<0.60", "[0.60, 0.80)", ">=0.80"]


def test_bootstrap_confidence_interval_returns_bounds():
    lower, upper = bootstrap_confidence_interval(
        lambda arr: float(np.mean(arr)),
        [0, 1, 0, 1],
        n_bootstraps=100,
        confidence=0.9,
        random_state=42,
    )
    assert lower <= upper


def test_summarise_uncertainty_creates_metrics():
    df = pd.DataFrame(
        {
            "__split__": ["train", "train", "test", "test"],
            "sex": ["m", "f", "m", "f"],
            "pred_sex": ["m", "f", "m", "f"],
            "pred_proba_m": [0.9, 0.1, 0.8, 0.2],
            "pred_proba_f": [0.1, 0.9, 0.2, 0.8],
        }
    )
    result = summarise_uncertainty(
        df,
        bootstrap_iterations=0,
        create_plot=False,
    )
    metrics = result["metrics"]
    assert set(["train", "test"]).issubset(metrics["split"].unique())
    annotated = result["annotated_predictions"]
    assert "max_confidence" in annotated.columns
    assert not result["calibration"].empty


def test_pairwise_uncertainty_summary():
    df = pd.DataFrame(
        {
            "pred_same_proba": [0.9, 0.2, 0.7, 0.4],
            "same_individual": [1, 0, 1, 0],
            "pair_id": ["a", "b", "c", "d"],
        }
    )
    result = summarise_pairwise_uncertainty(
        df,
        bootstrap_iterations=0,
    )
    assert not result["metrics"].empty
    assert "max_confidence" in result["annotated_predictions"].columns
