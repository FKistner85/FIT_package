import numpy as np
import pandas as pd
from FIT_python.general_pipeline_steps.dimensionality_reduction_wrapper import DimensionalityReducerTransformer

def test_none_method_dataframe():
    df = pd.DataFrame(np.random.randn(5, 3), columns=['a', 'b', 'c'])
    reducer = DimensionalityReducerTransformer(method=None)
    result = reducer.fit(df).transform(df)
    pd.testing.assert_frame_equal(result, df)
    assert reducer.get_feature_names_out() == ['a', 'b', 'c']

def test_none_method_array():
    arr = np.random.randn(4, 2)
    reducer = DimensionalityReducerTransformer(method=None)
    result = reducer.fit(arr).transform(arr)
    assert np.array_equal(result, arr)
    assert reducer.get_feature_names_out() == ['x0', 'x1']
