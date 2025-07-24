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
