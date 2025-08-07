import importlib
import pandas as pd
from FIT_python import config


def test_metadata_overrides(monkeypatch):
    monkeypatch.setitem(
        config.CONFIG["general_pipeline_steps"],
        "metadata_cols",
        ["keep_me"],
    )
    dim_mod = importlib.reload(
        importlib.import_module(
            "FIT_python.general_pipeline_steps.dimensionality_reduction_wrapper"
        )
    )
    sel_mod = importlib.reload(
        importlib.import_module(
            "FIT_python.general_pipeline_steps.feature_selection_wrapper"
        )
    )
    df = pd.DataFrame({"a": [1.0, 2.0], "keep_me": [10.0, 20.0]})
    reducer = dim_mod.DimensionalityReducerTransformer(method="pca", n_components=1)
    red = reducer.fit_transform(df)
    assert list(red.columns) == ["keep_me", "PCA1"]
    selector = sel_mod.FeatureSelectionTransformer(method="forward", k=1)
    out = selector.fit_transform(df, [0, 1])
    assert "keep_me" in out.columns
