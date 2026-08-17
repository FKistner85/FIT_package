"""Tests for subsample-size sweep and footprint LOO effect analysis."""

import numpy as np
import pandas as pd
import pytest

from FIT_python.pipeline_individual_id.evaluation import (
    evaluate_by_subsample_size,
)
from FIT_python.pipeline_individual_id.generate_trails_and_trailpairs import (
    loo_footprint_effect,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_results_df(trail_pairs, same_labels, preds):
    """Build a minimal results_df compatible with evaluate_by_subsample_size."""
    rows = []
    for (ta, tb), same, pred in zip(trail_pairs, same_labels, preds):
        rows.append(
            {
                "trail_a_id": ta,
                "trail_b_id": tb,
                "same_individual": same,
                "pred": pred,
            }
        )
    return pd.DataFrame(rows)


def _make_population_df():
    """Return a small two-individual dataset suitable for LOO tests."""
    rows = []
    for ind in ["A", "B"]:
        for i in range(4):
            rows.append(
                {
                    "id": f"{ind}_{i}",
                    "individual_id": ind,
                    "f1": float(i),
                    "f2": float(i) * 0.5,
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Tests: evaluate_by_subsample_size
# ---------------------------------------------------------------------------

class TestEvaluateBySubsampleSize:
    """Tests for evaluate_by_subsample_size."""

    def _make_df_for_sizes(self, sizes):
        """Create a results_df with trail names for the given subsample sizes."""
        trail_pairs = []
        same_labels = []
        preds = []
        for ss in sizes:
            ta = f"A_9_1_sub{ss}_sample1"
            tb = f"B_9_1_sub{ss}_sample1"
            trail_pairs.append((ta, tb))
            same_labels.append(False)
            preds.append(False)
        return _make_results_df(trail_pairs, same_labels, preds)

    def test_returns_dataframe(self):
        df = self._make_df_for_sizes([3, 5])
        out = evaluate_by_subsample_size(df)
        assert isinstance(out, pd.DataFrame)

    def test_expected_columns(self):
        df = self._make_df_for_sizes([3, 5])
        out = evaluate_by_subsample_size(df)
        assert set(out.columns) == {"subsample_size", "BCR", "TPR", "TNR", "n_pairs"}

    def test_one_row_per_size(self):
        df = self._make_df_for_sizes([1, 3, 5, 7, 9])
        out = evaluate_by_subsample_size(df)
        assert len(out) == 5

    def test_sorted_by_subsample_size(self):
        df = self._make_df_for_sizes([9, 1, 5])
        out = evaluate_by_subsample_size(df)
        sizes = out["subsample_size"].tolist()
        assert sizes == sorted(sizes)

    def test_n_pairs_counts(self):
        # Two pairs each at size 3 and one pair at size 5
        trail_pairs = [
            ("A_9_1_sub3_sample1", "B_9_1_sub3_sample1"),
            ("A_9_1_sub3_sample2", "B_9_1_sub3_sample2"),
            ("A_9_1_sub5_sample1", "B_9_1_sub5_sample1"),
        ]
        same_labels = [False, False, False]
        preds = [False, False, False]
        df = _make_results_df(trail_pairs, same_labels, preds)
        out = evaluate_by_subsample_size(df)
        row3 = out[out["subsample_size"] == 3]
        row5 = out[out["subsample_size"] == 5]
        assert row3["n_pairs"].values[0] == 2
        assert row5["n_pairs"].values[0] == 1

    def test_perfect_classification_gives_bcr_1(self):
        """When all predictions are correct BCR should be 1.0."""
        trail_pairs = [
            ("A_9_1_sub5_sample1", "B_9_1_sub5_sample1"),  # diff individual, pred diff
            ("A_9_1_sub5_sample2", "A_9_1_sub5_sample3"),  # same individual, pred same
        ]
        same_labels = [False, True]
        preds = [False, True]
        df = _make_results_df(trail_pairs, same_labels, preds)
        out = evaluate_by_subsample_size(df)
        assert pytest.approx(out["BCR"].iloc[0], abs=1e-6) == 1.0

    def test_trails_without_sub_pattern_grouped_separately(self):
        """Rows without _sub in trail name should appear in a NaN-key group."""
        trail_pairs = [
            ("A_trail1", "B_trail1"),
        ]
        df = _make_results_df(trail_pairs, [False], [False])
        out = evaluate_by_subsample_size(df)
        assert out["subsample_size"].isna().any()


# ---------------------------------------------------------------------------
# Tests: loo_footprint_effect
# ---------------------------------------------------------------------------

class _MockPipeline:
    """Minimal pipeline mock that predicts same_individual based on ind_a==ind_b."""

    def predict(self, comparisons):
        rows = []
        for c in comparisons:
            same = c["ind_a"] == c["ind_b"]
            rows.append({"same_individual": same, "pred": same})
        return pd.DataFrame(rows)


class TestLooFootprintEffect:
    """Tests for loo_footprint_effect."""

    def test_returns_dataframe(self):
        df = _make_population_df()
        pipeline = _MockPipeline()
        trail_ids = df[df["individual_id"] == "A"]["id"].tolist()
        out = loo_footprint_effect(trail_ids, df, pipeline)
        assert isinstance(out, pd.DataFrame)

    def test_expected_columns(self):
        df = _make_population_df()
        pipeline = _MockPipeline()
        trail_ids = df[df["individual_id"] == "A"]["id"].tolist()
        out = loo_footprint_effect(trail_ids, df, pipeline)
        assert set(out.columns) == {"footprint_id", "bcr", "delta_bcr"}

    def test_one_row_per_footprint(self):
        df = _make_population_df()
        pipeline = _MockPipeline()
        trail_ids = df[df["individual_id"] == "A"]["id"].tolist()
        out = loo_footprint_effect(trail_ids, df, pipeline)
        assert len(out) == len(trail_ids)

    def test_sorted_descending_by_delta_bcr(self):
        df = _make_population_df()
        pipeline = _MockPipeline()
        trail_ids = df[df["individual_id"] == "A"]["id"].tolist()
        out = loo_footprint_effect(trail_ids, df, pipeline)
        deltas = out["delta_bcr"].tolist()
        assert deltas == sorted(deltas, reverse=True)

    def test_accepts_precomputed_full_bcr(self):
        df = _make_population_df()
        pipeline = _MockPipeline()
        trail_ids = df[df["individual_id"] == "A"]["id"].tolist()
        out = loo_footprint_effect(trail_ids, df, pipeline, full_bcr=1.0)
        # For each row: delta_bcr == 1.0 - bcr when bcr is not NaN,
        # or delta_bcr is NaN when bcr is NaN (1.0 - NaN = NaN)
        for _, row in out.iterrows():
            if pd.isna(row["bcr"]):
                assert pd.isna(row["delta_bcr"])
            else:
                assert row["delta_bcr"] == pytest.approx(1.0 - row["bcr"], abs=1e-9)

    def test_footprint_ids_are_preserved(self):
        df = _make_population_df()
        pipeline = _MockPipeline()
        trail_ids = df[df["individual_id"] == "A"]["id"].tolist()
        out = loo_footprint_effect(trail_ids, df, pipeline)
        assert set(out["footprint_id"]) == set(trail_ids)
