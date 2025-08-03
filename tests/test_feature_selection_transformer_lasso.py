import pandas as pd
from FIT_python.general_pipeline_steps.feature_selection_wrapper import FeatureSelectionTransformer


def test_lasso_handles_string_labels():
    X = pd.DataFrame({"a": [1, 2, 3, 4, 5, 6], "b": [2, 3, 4, 5, 6, 7]})
    y = ["f", "m", "f", "m", "f", "m"]
    selector = FeatureSelectionTransformer(method="lasso", k=1)
    selector.fit(X, y)
    assert len(selector.selected_features_) == 1
