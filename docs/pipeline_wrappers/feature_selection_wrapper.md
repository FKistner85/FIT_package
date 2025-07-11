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
【F:src/FIT_python/pipeline_general/feature_selection_wrapper.py†L60-L71】

* **Forward Selection** erweitert das Modell schrittweise um diejenigen Features, die den größten Zugewinn an Erklärungsstärke liefern. Das Verfahren ist leicht verständlich, kann aber zu suboptimalen Kombinationen führen, da früh getroffene Entscheidungen nicht revidiert werden.

* **Random Forest Importance** ordnet die Merkmale nach ihrer Bedeutsamkeit in einem Ensemble aus Entscheidungsbäumen. Es kann auch nichtlineare Zusammenhänge erkennen. Bei kleinen Stichproben schwanken die Importances jedoch stark und korrelierte Variablen werden bevorzugt.

* **Variance Threshold** entfernt Spalten mit sehr geringer Streuung. Das ist schnell und komplett unbeaufsichtigt, riskiert aber, seltene aber aussagekräftige Merkmale auszuschließen.

* **Univariate Tests** wie der ANOVA-F-Score untersuchen jede Variable für sich. Diese Einfachheit macht die Methode erklärbar, lässt aber Wechselwirkungen zwischen Features außer Acht.

* **LASSO** nutzt eine \(\ell_1\)-Strafung und erzeugt dadurch spärliche Koeffizienten. Es eignet sich für hochdimensionale Probleme, kann allerdings wichtige Prädiktoren unterschätzen, wenn sie stark korrelieren.

Wird `method=None` gewählt, verbleiben alle Features im Datensatz.

### Referenzen
* Breiman, L. (2001). "Random Forests." *Machine Learning*.
* Draper, N., & Smith, H. (1966). *Applied Regression Analysis*. Wiley.
* Fisher, R. A. (1925). *Statistical Methods for Research Workers*.
* Tibshirani, R. (1996). "Regression shrinkage and selection via the lasso." *Journal of the Royal Statistical Society, Series B*.
