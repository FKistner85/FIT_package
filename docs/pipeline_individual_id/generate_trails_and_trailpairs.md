# generate_trails_and_trailpairs.py

`generate_pairwise_comparisons_from_df()` constructs training pairs of trails. Its docstring details the procedure:

```python
    1) Erzeuge group_id = individual_id, bzw. wenn NaN/unknown dann fallback trail.
    2) Sample pro group_id nicht-überlappende Chunks der Länge chunk_size,
       daraus bis zu max_trails_per_animal Trails jeder Länge in trail_size_list.
    3) Baue alle Cross-Individual-Paare UND alle Within-Individual, cross-chunk Paare.
    4) same_individual = True/False, oder "unknown" wenn eine Seite fallback benutzt.
    5) same_sex = True/False/"unknown" analog.
    6) StratifiedKFold nach trail_size_a.
    7) Summary-Tabelle mit pro-Länge und Total-Zeile inkl. avg/sd Pair counts.
```
【F:src/FIT_python/pipeline_individual_id/generate_trails_and_trailpairs.py†L34-L42】

The function samples fixed-length segments from each individual, creates both cross- and within-individual pairings and assigns folds using `StratifiedKFold`. A summary table counts the resulting pairs.
