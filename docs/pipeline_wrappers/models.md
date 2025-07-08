# models.py

`MODELS` enthält vorbereitete Instanzen verschiedener Klassifikatoren:

```python
    MODELS = {
        "logreg_l2": LogisticRegression(...),
        "rf_small":  RandomForestClassifier(...),
        "svm_rbf":   SVC(kernel="rbf", ...),
        "lda":       LinearDiscriminantAnalysis(),
        ...
    }
```
【F:src/FIT_python/pipeline_sex/models.py†L10-L31】

Die Sammlung ermöglicht schnelle Vergleiche zwischen linearen und nichtlinearen Verfahren. Ensemble-Methoden wie Random Forest sind robust gegen Ausreißer, während SVMs gute Trennung bei kleinen Datensätzen bieten. LDA liefert lineare Entscheidungsgrenzen und ist die Standardwahl der Pipeline.
