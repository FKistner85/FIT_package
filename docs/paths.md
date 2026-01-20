# Results Directory Structure

The helper function [`get_species_paths`](../src/FIT_python/utils/paths.py) defines the canonical
output layout for a species. A ``section`` must be provided to indicate
which part of the project the results belong to. Available sections are
``dataprocessing``, ``sex_modelling`` and ``individual_id``. The selected
section determines the root folder.

**Important:** `EXPERIMENT_ROOT` (e.g., `results/fit_notebook_final/`) already serves as the results root directory. There is no additional `results/` subdirectory to avoid double nesting like `results/results/`.

For a species within a given section the following directories are created:

* ``dataprocessing``

```
<experiment_root>/<section>/<species>/
    figures/
    tables/
```

* ``sex_modelling`` and ``individual_id``

```
<experiment_root>/<section>/<species>/
    models/<metric>/<species>.joblib
    figures/
    tables/
```

Example for the `fit_notebook_final` experiment:
```
results/fit_notebook_final/
    data/
        cleaned/
        splits/
    dataprocessing/
        eurasian_otter/
            figures/
            tables/
    sex_modelling/
        eurasian_otter/
            models/
                balanced_accuracy/
            figures/
            tables/
    individual_id/
        eurasian_otter/
            master_pairs.csv
            figures/
            tables/
```

Use `get_species_paths` whenever code needs access to these paths to keep the project structure consistent.
