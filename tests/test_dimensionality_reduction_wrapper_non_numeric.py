import pandas as pd
import numpy as np
from FIT_python.general_pipeline_steps.dimensionality_reduction_wrapper import DimensionalityReducerTransformer

def test_non_numeric_columns_ignored():
    df = pd.DataFrame({'a':[1.0,2.0],'b':[3.0,4.0],'id':['x','y']})
    reducer = DimensionalityReducerTransformer(method='pca', n_components=1)
    result = reducer.fit_transform(df)
    assert list(result.columns) == ['id', 'PCA1']
    assert result.shape == (2, 2)
    assert reducer.get_feature_names_out() == ['PCA1']
