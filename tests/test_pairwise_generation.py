import pandas as pd
from FIT_python.pipeline_individual_id.generate_trails_and_trailpairs import generate_pairwise_comparisons_from_df


def test_pairwise_generation_evaluation_mode():
    df = pd.DataFrame({
        "id": [1, 2, 3, 4],
        "individual_id": ["A", "A", "B", "B"],
        "trail": ["A_1", "A_1", "B_1", "B_1"],
        "sex": ["f", "f", "m", "m"],
    })

    comps, summary = generate_pairwise_comparisons_from_df(
        df,
        strategy="select",
        evaluation=True,
    )

    assert len(comps) == 1
    comp = comps[0]
    assert comp["fold_a"] is None
    assert comp["fold_b"] is None
    assert comp["same_fold"] is None
    assert comp["fold"] is None
    assert summary["n_pairs"].iloc[0] == 1


def test_pairwise_generation_filters_na_and_unknown_trails():
    """Test that NA and 'unknown' trail values are filtered before pair generation."""
    df = pd.DataFrame({
        "id": [1, 2, 3, 4, 5, 6, 7, 8],
        "individual_id": ["A", "A", "B", "B", "C", "C", "D", "D"],
        "trail": ["A_1", "A_1", None, "B_1", "unknown", "C_1", "UNKNOWN", "D_1"],
        "sex": ["f", "f", "m", "m", "f", "f", "m", "m"],
    })

    comps, summary = generate_pairwise_comparisons_from_df(
        df,
        strategy="select",
        evaluation=True,
    )

    # Only trails A_1, B_1, C_1, D_1 should be included (4 trails)
    # Number of pairs = C(4, 2) = 6
    assert summary["n_trails"].iloc[0] == 4
    
    # Verify no comparison includes NA or unknown trails
    trail_ids = set()
    for comp in comps:
        trail_ids.add(comp["trail_a_id"])
        trail_ids.add(comp["trail_b_id"])
    
    assert None not in trail_ids
    assert "unknown" not in trail_ids
    assert "UNKNOWN" not in trail_ids
