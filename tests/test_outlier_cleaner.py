import pandas as pd
import numpy as np

# Import config first to avoid circular import issues in tests
import FIT_python.config  # noqa: F401
from FIT_python.general_pipeline_steps.outlier_wrapper import OutlierCleanerTransformer


def test_outlier_cleaner_handles_mixed_dataframe_clip():
    df = pd.DataFrame({
        'num1': [1, 2, 100],
        'num2': [5.0, -10.0, 0.5],
        'txt': ['a', 'b', 'c'],
    })
    trans = OutlierCleanerTransformer(method='clip', lower_quantile=0.0, upper_quantile=1.0)
    trans.fit(df)
    assert trans.numeric_cols_ == ['num1', 'num2']
    result = trans.transform(df)
    assert list(result['txt']) == ['a', 'b', 'c']
    pd.testing.assert_frame_equal(
        result[['num1', 'num2']], df[['num1', 'num2']], check_dtype=False
    )


def test_outlier_cleaner_handles_mixed_dataframe_zscore():
    df = pd.DataFrame({
        'num1': [1.0, 2.0, 100.0],
        'num2': [5.0, -10.0, 0.5],
        'txt': ['x', 'y', 'z'],
    })
    trans = OutlierCleanerTransformer(method='zscore', z_thresh=10.0)
    trans.fit(df)
    assert trans.numeric_cols_ == ['num1', 'num2']
    result = trans.transform(df)
    assert list(result['txt']) == ['x', 'y', 'z']
    pd.testing.assert_frame_equal(
        result[['num1', 'num2']], df[['num1', 'num2']], check_dtype=False
    )

