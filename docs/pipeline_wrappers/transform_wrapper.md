# transform_wrapper.py

Der `NumericTransformer` bereitet die Rohdaten für das Modell vor. Die wichtigsten Schritte sind in der Klasse dokumentiert:

```python
    Reiner Numeric-Transformer:
      1) Komma → Punkt in allen Feature-Spalten
      2) Coercion zu float (non-convertible → NaN)
      3) Liefert reines NumPy-Array X zurück.
```
【F:src/FIT_python/pipeline_sex/transform_wrapper.py†L12-L15】

So bleiben Zielspalten unverändert und die Pipeline erhält eine konsistente numerische Matrix.
