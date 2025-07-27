import numpy as np
import pytest
from FIT_python.general_pipeline_steps.dimensionality_reduction_wrapper import DimensionalityReducerTransformer


def test_umap_requires_y():
    X = np.random.randn(5, 3)
    reducer = DimensionalityReducerTransformer(method="umap", n_components=2)
    with pytest.raises(ValueError):
        reducer.fit(X)

