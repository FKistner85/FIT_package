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

Ersteres wandelt Strings in Fließzahlen um, letzteres erzeugt binäre Zielvektoren für mehrklassige Aufgaben.
