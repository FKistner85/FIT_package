# outlier_wrapper.py

`OutlierCleanerTransformer` mitigates extreme values by either clipping percentiles or applying a Z-score threshold:

```python
    Outlier-Bereinigung durch Clipping oder Z-Score-Begrenzung.

    Methoden:
      - 'clip': alle Features auf [q_low, q_high] clippen (Percentile-Clipping)
      - 'zscore': Werte außerhalb von ±z_thresh*σ auf ±z_thresh*σ setzen (Winsorizing)
```
【F:src/FIT_python/pipeline_general/outlier_wrapper.py†L8-L15】

Clipping is a simple non-parametric technique that replaces extreme values by upper and lower quantiles. It is robust and easy to explain, yet it risks truncating genuinely informative observations if they naturally fall outside the chosen bounds.

Z-score limiting (also called winsorising) scales values by their standard deviation and caps them at a multiple of the estimated spread. This retains the overall shape of the distribution but implicitly assumes approximate normality. If the feature distribution is strongly skewed, the resulting bounds may still be inappropriate.

Both methods operate column-wise and the transformer preserves the input type (DataFrame or `ndarray`). Parameters allow custom quantile limits or z-score thresholds to tailor the cleaning to a specific dataset.

**References**
* Tukey, J. W. (1962). "The future of data analysis." *Annals of Mathematical Statistics*.
