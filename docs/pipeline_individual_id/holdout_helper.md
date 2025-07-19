# holdout_helper.py

`generate_holdout_sets()` collects sequential holdout splits for all species directories:

```python
    def generate_holdout_sets(
        split_dir: Path,
        val_sizes: Sequence[int],
        iterations: int,
        seed: int = 42,
    ) -> dict[str, list[dict[str, pd.DataFrame]]]:
        """Return sequential holdout splits for every species.
        ...
        Mapping from species codes to lists of split dictionaries. Each
        dictionary contains ``train_df`` and ``val_df`` along with
        ``iteration`` and ``n_val`` information.
        """
```
【F:src/FIT_python/pipeline_individual_id/holdout_helper.py†L8-L37】

The helper iterates over the provided `split_dir`, loads each `train.parquet` and
uses `sequential_holdout_ids` to derive train/validation splits. Returned
DataFrames are ready for further processing.
