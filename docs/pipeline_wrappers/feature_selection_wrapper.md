# feature_selection_wrapper.py

`FeatureSelectionTransformer` kann unterschiedliche Auswahlkriterien einsetzen. Im Konstruktor ist eine der folgenden Methoden erlaubt:

```python
    def __init__(
        self,
        method: str = None,  # 'forward', 'random_forest', 'variance', 'univariate', 'lasso', or None (use all)
        k: int = None,
        random_state: int = 0
    ):
        allowed_methods = [None, 'forward', 'random_forest', 'variance', 'univariate', 'lasso']
```
【F:src/FIT_python/pipeline_sex/feature_selection_wrapper.py†L61-L68】

*Forward selection* fügt Merkmale iterativ hinzu, *Random Forest* nutzt Importance-Werte, *Variance* filtert niedrige Streuung, *Univariate* wendet einzelne ANOVA-Tests an und *LASSO* setzt \(\ell_1\)-Regularisierung ein. Keine Methode (`None`) verwendet alle Features.
