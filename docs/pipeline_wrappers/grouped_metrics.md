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

* ``individual_accuracies`` liefert getrennte Genauigkeiten für jedes Individuum sowie den balancierten Mittelwert für Weibchen und Männchen. Dies ist hilfreich, um Verzerrungen bei ungleich verteilten Gruppen früh zu erkennen. Der Nachteil ist der höhere Rechenaufwand, wenn sehr viele Tiere vorliegen.

* ``individual_majority_stats`` zählt, bei wie vielen Individuen die Mehrzahl der Vorhersagen korrekt ist. Damit lässt sich nachvollziehen, ob einzelne Tiere systematisch falsch klassifiziert werden. Die Kennzahl ist grob, liefert aber einen schnellen Überblick.

### Referenzen
* Ausführliche Beispiele für gruppierte Metriken finden sich in der [scikit-learn Dokumentation zu aggregierten Scores](https://scikit-learn.org/stable/modules/model_evaluation.html).
