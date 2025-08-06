# Results Directory Structure

The helper function [`get_species_paths`](../src/FIT_python/utils/paths.py) defines the canonical output layout for a species. It creates a set of directories under the experiment results folder and should be treated as the single source of truth when reading from or writing to these locations.

```
results/<experiment>/results_data/<species>/
    models/<metric>/<species>.joblib
    predictions/<species>_all_predictions.csv
    heatmaps/
    logs/
    search/
```

Use `get_species_paths` whenever code needs access to these paths to keep the project structure consistent.
