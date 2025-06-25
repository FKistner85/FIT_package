# Step 2: Cleaning

After loading, textual target columns can be cleaned using
`sanitize_labels` from `FIT_python.data_import_utils`. The `DataImporter`
and `DataPipeline` classes wrap this function to fill missing values,
normalise strings and apply optional mappings.

**Key Functions**
- `sanitize_labels`
- `DataImporter.clean`
- `DataPipeline.clean_targets`

**Algorithmic Cost**
- Linear in the number of values processed.

