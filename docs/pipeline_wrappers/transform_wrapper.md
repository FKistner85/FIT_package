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

Der Transformer eignet sich für Datensätze, die ursprünglich gemischte Typen enthalten, etwa durch Exporte aus Tabellenkalkulationen. Durch die strikte Umwandlung in Gleitkommazahlen werden Fehlwerte als ``NaN`` markiert und können anschließend imputiert werden. Ein Nachteil besteht darin, dass eventuelle Kategorieinformationen verloren gehen, falls sie nicht vorher kodiert wurden.

### Referenzen
* Weitere Hintergründe zum Umgang mit fehlenden und numerischen Daten finden sich in der [pandas Dokumentation](https://pandas.pydata.org/).
