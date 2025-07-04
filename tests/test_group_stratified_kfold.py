import pandas as pd
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from FIT_python.split_utils import group_stratified_kfold


def test_dynamic_folds_balanced():
    data = {
        "individual_id": [1, 2, 3, 4],
        "sex": ["F", "F", "M", "M"],
    }
    df = pd.DataFrame(data)
    result = group_stratified_kfold(df, n_splits=3)
    assert result["Fold"].nunique() == 2
    for fold in result["Fold"].unique():
        classes = set(result[result["Fold"] == fold]["sex"])
        assert len(classes) > 1


def test_too_few_groups_error():
    data = {
        "individual_id": [1, 2, 3, 4],
        "sex": ["F", "F", "F", "M"],
    }
    df = pd.DataFrame(data)
    with pytest.raises(ValueError):
        group_stratified_kfold(df, n_splits=5)
