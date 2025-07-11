# transform_utils.py

`convert_numeric` und `one_hot_encode_targets` vereinfachen die Datenvorbereitung:

```python
    def convert_numeric(df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
        """Convert feature columns to numeric and one-hot encode pure strings."""
        ...
    def one_hot_encode_targets(df: pd.DataFrame, target_cols: List[str]) -> Tuple[np.ndarray, Dict[str, List[str]]]:
        """One-hot encode target columns."""
```
【F:src/FIT_python/pipeline_sex/transform_utils.py†L9-L33】

* ``convert_numeric`` ersetzt Kommas durch Punkte, versucht eine numerische Konvertierung und kodiert rein kategoriale Spalten automatisch per One-Hot-Encoding. So können auch String-Spalten in Modellen genutzt werden.

* ``one_hot_encode_targets`` erstellt für jede Zielspalte eine binäre Kodierung und liefert neben dem Array auch eine Mapping-Tabelle zurück. Dies ist hilfreich bei Mehrklassenproblemen, führt aber zu breiteren Matrizen.

* ``save_target_mapping`` schreibt diese Zuordnung in eine JSON-Datei, um Vorhersagen später korrekt dekodieren zu können.

### Referenzen
* Die Funktionen orientieren sich an bewährten Mustern aus der [pandas](https://pandas.pydata.org/) und [scikit-learn](https://scikit-learn.org/) Dokumentation.
