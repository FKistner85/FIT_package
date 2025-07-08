# outlier_wrapper.py

`OutlierCleanerTransformer` kennt zwei Verfahren zur Dämpfung extremer Werte:

```python
    Outlier-Bereinigung durch Clipping oder Z-Score-Begrenzung.
      - 'clip': alle Features auf [q_low, q_high] clippen (Percentile-Clipping)
      - 'zscore': Werte außerhalb von ±z_thresh*σ auf ±z_thresh*σ setzen (Winsorizing)
```
【F:src/FIT_python/pipeline_sex/outlier_wrapper.py†L8-L14】

Clipping ignoriert die Extremwerte jenseits der gewählten Quantile, während Z-Score-Winsorizing die vorhandene Streuung beibehält, aber Normalverteilung der Merkmale voraussetzt.
