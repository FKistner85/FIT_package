# outlier_wrapper.py

`OutlierCleanerTransformer` mitigates extreme values by either clipping percentiles or applying a Z-score threshold:

```python
    Outlier-Bereinigung durch Clipping oder Z-Score-Begrenzung.

    Methoden:
      - 'clip': alle Features auf [q_low, q_high] clippen (Percentile-Clipping)
      - 'zscore': Werte außerhalb von ±z_thresh*σ auf ±z_thresh*σ setzen (Winsorizing)
```
【F:src/FIT_python/pipeline_individual_id/outlier_wrapper.py†L8-L14】

Clipping is robust to anomalies but may discard genuine variation. Z-score limiting keeps the data shape but assumes normality of feature values.
