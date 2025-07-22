# Display Mapping for Anonymous IDs

This page explains how to create a mapping between original identifiers and
anonymised names for plotting functions. It also covers the optional legend
parameter of the UMAP utilities and how dendrogram tick labels pick up the
colours of these names.

## Generating the mapping

Call `generate_display_mapping()` with the path to a CSV table containing at
least an ``individual_id`` column. The function collects all unique IDs,
assigns short anonymous labels (``A``/``B``/``C`` …) and returns a dictionary:

```python
from FIT_python.Visualisations.id_style import generate_display_mapping

mapping = generate_display_mapping(path_to_predictions)
# mapping == {"jj_12": "A", "jj_13": "B", ...}
```

Save the dictionary to disk so it can be reused across notebooks and pipelines.

## Applying before plotting

When visualising results, load the mapping and rename the ID column before
calling any plotting helper. Most plotting functions automatically pick up the
``individual_id`` column:

```python
from FIT_python.Visualisations.id_style import apply_display_mapping

results = apply_display_mapping(results, mapping)
plot_umap_by_individual(results, FIG_DIR / "umap.png", legend=True)
```

Any dataframe containing a column named ``individual_id`` or ``trail`` can be
passed through `apply_display_mapping()` to replace the raw identifiers with the
anonymous labels.

## Legends in UMAP plots

Functions such as `plot_umap_scatter` and `plot_umap_by_individual` accept a
``legend`` flag. Set ``legend=False`` to hide the legend when layering multiple
plots or when the labels are obvious from context. The default is ``True`` which
shows the legend beside the scatter plot.

## Dendrogram tick colours

The helper `plot_dendrogram` sets the tick label colour based on the anonymised
``individual_id``. Labels fall back to black when an ID is missing from the
mapping. This colouring makes it easier to follow the branches of each
individual in the dendrogram.
