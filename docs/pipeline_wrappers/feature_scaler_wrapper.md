# feature_scaler_wrapper.py

Der `FeatureScalerTransformer` stellt zwei Varianten des Skalierens bereit:

```python
    Scaler für numerische Features, mit zwei Modi:
      - method='standard': StandardScaler (z-Transformation)
      - method='robust':   RobustScaler (Median & IQR)
```
【F:src/FIT_python/pipeline_sex/feature_scaler_wrapper.py†L10-L14】

*StandardScaler* setzt Mittelwert 0 und Varianz 1 voraus und reagiert empfindlich auf Ausreißer. *RobustScaler* nutzt Median und Interquartilsabstand und ist stabiler bei Extremwerten, kann aber skalengetreue Merkmale verzerren.
