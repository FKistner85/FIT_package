# grouped_metrics.py

Dieses Modul enthält Hilfsfunktionen zur Bewertung der Modellgüte auf Individuen-Ebene. Beispielhaft `individual_accuracies` berechnet pro Tier den Anteil korrekter Vorhersagen:

```python
    def individual_accuracies(y_true, y_pred, ids):
        df = pd.DataFrame({'y_true': y_true, 'y_pred': y_pred, 'id': ids})
        # FutureWarning vermeiden mit include_groups=False
        acc_per = df.groupby('id').apply(
            lambda g: (g.y_true == g.y_pred).mean(),
            include_groups=False
        )
```
【F:src/FIT_python/pipeline_sex/grouped_metrics.py†L3-L8】

Die Funktionen helfen dabei, Balance zwischen männlichen und weiblichen Individuen zu beurteilen.
