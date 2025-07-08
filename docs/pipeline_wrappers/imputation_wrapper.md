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

Der Ansatz kann komplexe Abhängigkeiten zwischen Features abbilden und produziert plausible Werte, erfordert aber mehr Rechenzeit als einfache Strategien wie Mittelwert-Imputation.
