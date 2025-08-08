# Results Directory Structure

The helper function [`get_species_paths`](../src/FIT_python/utils/paths.py) defines the canonical
output layout for a species. A ``section`` must be provided to indicate
which part of the project the results belong to. Available sections are
``dataprocessing``, ``sex_modelling`` and ``individual_id``. The selected
section determines the root folder under ``results``.

For a species within a given section the following directories are created:

* ``dataprocessing``

```
results/<section>/<species>/
    figures/
    tables/
```

* ``sex_modelling`` and ``individual_id``

```
results/<section>/<species>/
    models/<metric>/<species>.joblib
    figures/
    tables/
```

Use `get_species_paths` whenever code needs access to these paths to keep the project structure consistent.
