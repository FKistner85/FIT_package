# outlier_wrapper.py

`OutlierCleanerTransformer` kennt zwei Verfahren zur Dämpfung extremer Werte:

```python
    Outlier-Bereinigung durch Clipping oder Z-Score-Begrenzung.
      - 'clip': alle Features auf [q_low, q_high] clippen (Percentile-Clipping)
      - 'zscore': Werte außerhalb von ±z_thresh*σ auf ±z_thresh*σ setzen (Winsorizing)
```
【F:src/FIT_python/pipeline_general/outlier_wrapper.py†L8-L15】

* **Clippen** ersetzt Werte oberhalb bzw. unterhalb festgelegter Quantile durch die jeweiligen Grenzwerte. Das Vorgehen ist robust und unkompliziert, kann jedoch wirkliche Extremfälle verdecken.

* **Z-Score-Winsorizing** setzt Werte außerhalb eines Vielfachen der Standardabweichung auf eben diese Grenze. Dadurch bleibt die Form der Verteilung erhalten, vorausgesetzt sie ist annähernd normalverteilt. Bei stark schiefen Merkmalen wählt man die Schwelle besser vorsichtig.

### Referenzen
* Tukey, J. W. (1962). "The future of data analysis." *Annals of Mathematical Statistics*.
