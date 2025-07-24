import pytest
pytest.importorskip("pandas")
import pandas as pd
from FIT_python.data_split_and_summary.data_import_utils import get_feature_cols


def test_get_feature_cols_prefix_detection():
    df = pd.DataFrame(
        {
            "id": [1, 2],
            "v1": [0.0, 0.1],
            "AreaWidth": [1.0, 2.0],
            "DistMean": [0.2, 0.3],
            "ANGdeg": [5, 6],
            "t12": [0.4, 0.5],
            "extra": [7, 8],
        }
    )
    assert get_feature_cols(df) == ["v1", "AreaWidth", "DistMean", "ANGdeg", "t12"]


def test_get_feature_cols_default_branch():
    df = pd.DataFrame(
        {
            "a": ["x", "y"],
            "b": ["x", "y"],
            "c": ["x", "y"],
            "d": ["x", "y"],
            "e": ["x", "y"],
            "num1": [1.0, 2.0],
            "num2": [3, 4],
        }
    )
    assert get_feature_cols(df) == ["num1", "num2"]
