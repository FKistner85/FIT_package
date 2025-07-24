import pytest
pytest.importorskip("pandas")
import pandas as pd
from FIT_python.Visualisations.display_mapping import generate_display_mapping, apply_display_mapping


def test_generate_and_apply_display_mapping():
    df = pd.DataFrame(
        {
            "individual_id": ["A", "B", "A"],
            "trail": ["t1", "t2", "t3"],
        }
    )
    mapping = generate_display_mapping(df)
    expected = {
        "A": "Ind_1",
        "B": "Ind_2",
        "t1": "Ind_1_trail_orig_1",
        "t3": "Ind_1_trail_orig_2",
        "t2": "Ind_2_trail_orig_1",
    }
    assert mapping == expected

    out = apply_display_mapping(df, mapping)
    assert out["display_id"].tolist() == ["Ind_1", "Ind_2", "Ind_1"]
    assert out["display_trail"].tolist() == [
        "Ind_1_trail_orig_1",
        "Ind_2_trail_orig_1",
        "Ind_1_trail_orig_2",
    ]
