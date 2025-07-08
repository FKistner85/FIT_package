# imputation_wrapper.py

Der `ImputationWrapper` nutzt einen `IterativeImputer` mit Random-Forest-Regressor:

```python
    self.imputer = IterativeImputer(
        estimator=RandomForestRegressor(
            n_estimators=n_estimators,
            random_state=random_state
        ),
        max_iter=max_iter,
        initial_strategy='median',
        random_state=random_state
    )
```
【F:src/FIT_python/pipeline_sex/imputation_wrapper.py†L17-L26】

Der Ansatz kann komplexe Abhängigkeiten zwischen Features abbilden und liefert dadurch oft realistischere Werte als simple Mittelwert- oder Medianfüllungen. Besonders bei stark korrelierten Variablen spielt der Random-Forest-Schätzer seine Stärken aus. Nachteilig sind der höhere Speicherbedarf und eine längere Laufzeit, weshalb sich die Methode vor allem für endgültige Analysen eignet.

### Referenzen
* Stekhoven, D. J., & Bühlmann, P. (2012). "MissForest—non-parametric missing value imputation for mixed-type data." *Bioinformatics*.
* Die Implementierung basiert auf dem [IterativeImputer von scikit-learn](https://scikit-learn.org/stable/modules/impute.html#iterative-imputer).
