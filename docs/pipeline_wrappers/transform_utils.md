# transform_utils.py

`convert_numeric` und `one_hot_encode_targets` vereinfachen die Datenvorbereitung:

```python
    def convert_numeric(df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
        """Convert feature columns to float, replacing comma decimal separators."""
        ...
    def one_hot_encode_targets(df: pd.DataFrame, target_cols: List[str]) -> Tuple[np.ndarray, Dict[str, List[str]]]:
        """One-hot encode target columns."""
```
【F:src/FIT_python/pipeline_sex/transform_utils.py†L9-L33】

* ``convert_numeric`` bereitet gemischte Datensätze auf, indem Kommas in Dezimalpunkten ersetzt und nicht konvertierbare Werte zu ``NaN`` werden. Das erleichtert die anschließende numerische Verarbeitung, birgt jedoch die Gefahr, dass unerkannte Fehler in den Ursprungsdaten verdeckt werden.

* ``one_hot_encode_targets`` erstellt für jede Zielspalte eine binäre Kodierung und liefert neben dem Array auch eine Mapping-Tabelle zurück. Dies ist hilfreich bei Mehrklassenproblemen, führt aber zu breiteren Matrizen.

* ``save_target_mapping`` schreibt diese Zuordnung in eine JSON-Datei, um Vorhersagen später korrekt dekodieren zu können.

### Referenzen
* Die Funktionen orientieren sich an bewährten Mustern aus der [pandas](https://pandas.pydata.org/) und [scikit-learn](https://scikit-learn.org/) Dokumentation.
