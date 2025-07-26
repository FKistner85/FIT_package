# feature_scaler_wrapper.py

Der `FeatureScalerTransformer` stellt zwei Varianten des Skalierens bereit:

```python
    Scaler für numerische Features, mit zwei Modi:
      - method='standard': StandardScaler (z-Transformation)
      - method='robust':   RobustScaler (Median & IQR)
```
【F:src/FIT_python/pipeline_sex/feature_scaler_wrapper.py†L10-L14】

* **StandardScaler** setzt Mittelwert 0 und Varianz 1 voraus. Er eignet sich für Modelle, die von normalverteilten Merkmalen ausgehen oder Distanzmaße verwenden. Vorteilhaft ist die weit verbreitete Unterstützung in vielen Algorithmen; störend kann der Einfluss einzelner Ausreißer sein.

* **RobustScaler** verwendet Median und Interquartilsabstand und ist damit unempfindlicher gegenüber Ausreißern. Dies bietet sich bei schiefen oder verrauschten Daten an. Der Preis dafür ist eine gewisse Verzerrung bei eigentlich skalengetreuen Variablen.

Nicht-numerische Spalten im DataFrame werden unverändert durchgereicht.

### Referenzen
* Die [scikit-learn Dokumentation zur Vorverarbeitung](https://scikit-learn.org/stable/modules/preprocessing.html#scaling-features) erläutert die Skalierer im Detail.
