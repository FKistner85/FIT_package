# rcv_sampling.py

`generate_rcv()` builds the Recaptured Control Variation dataset by removing the indices used in the current comparison:

```python
    def generate_rcv(full_df: pd.DataFrame, exclude_indices: list) -> pd.DataFrame:
        """Create the **R**ecaptured **C**ontrol **V**ariation dataset.
        ...
        Subset of ``full_df`` excluding ``exclude_indices`` and with
        ``'individual_id'`` and ``'Trail'`` set to ``"RCV"``.
```
【F:src/FIT_python/pipeline_individual_id/rcv_sampling.py†L13-L29】

The resulting subset serves as a reference space into which pairwise trails are projected.
