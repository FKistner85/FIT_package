# feature_scaler_wrapper.py

The `FeatureScalerTransformer` applies either standard or robust scaling. The options are summarised in its docstring:

```python
    Scaler für numerische Features, mit zwei Modi:
      - method='standard': StandardScaler (z-Transformation)
      - method='robust':   RobustScaler (Median & IQR)
```
【F:src/FIT_python/pipeline_individual_id/feature_scaler_wrapper.py†L10-L14】

* **StandardScaler** rescales features to zero mean and unit variance. It assumes a roughly Gaussian distribution and can be sensitive to outliers.
* **RobustScaler** uses median and interquartile range, making it more stable for heavy‑tailed data but potentially distorting well-behaved variables.
