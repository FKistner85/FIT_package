# feature_scaler_wrapper.py

The `FeatureScalerTransformer` applies either standard or robust scaling. The options are summarised in its docstring:

```python
    Scaler für numerische Features, mit zwei Modi:
      - method='standard': StandardScaler (z-Transformation)
      - method='robust':   RobustScaler (Median & IQR)
```
【F:src/FIT_python/pipeline_individual_id/feature_scaler_wrapper.py†L10-L14】

* **StandardScaler** rescales features to zero mean and unit variance. This is the classic z‑transformation often recommended when the data roughly follow a normal distribution. In practice it is widely used for algorithms that rely on gradient descent or distance based measures. The downside is that it can be heavily influenced by extreme outliers.

* **RobustScaler** uses the median and the inter‑quartile range instead of the mean and standard deviation. This makes the scaling procedure much less sensitive to extreme values and heavy‑tailed distributions. The approach follows the ideas discussed for robust statistics in texts such as *Huber, 1981*. While it offers stability for messy data, it may distort variables that are already well behaved.

The transformer returns a `pandas.DataFrame` if the input was a DataFrame and otherwise an `ndarray`, so it integrates neatly into scikit‑learn pipelines.

**References**
* Huber, P. J. (1981). *Robust Statistics*. John Wiley & Sons.
* The [scikit‑learn preprocessing documentation](https://scikit-learn.org/stable/modules/preprocessing.html#scaling-features) provides additional background.
