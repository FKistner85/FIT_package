# geometric_pairwise_projection.py

`run_all_pairwise_projections_parallel()` performs the heavy computation for comparing trail pairs. Its docstring summarises the steps:

```python
    Für jede Paarung:
      0) Falls use_sexmodel_prediction, Sex-Modell laden & predict_proba auf ganzem df_base vorberechnen
      1) Basis-DF bereinigen
      2) Feature-Selection (einmal mit k_max)
      3) RCV-Set als Komplement der Indizes
      4) predict_proba für A, B, R extrahieren und mitteln (jeweils avg für 0/1)
      5) Für jede (reducer, n_components, k):
         - DimRed erzeugen
         - Abstände berechnen
         - Result-Dict inkl. avg_proba_A_0/1, avg_proba_B_0/1, avg_proba_R_0/1
```
【F:src/FIT_python/pipeline_individual_id/geometric_pairwise_projection.py†L155-L164】

The function iterates over reduction methods (LDA, PCA, UMAP) and different numbers of components. Distances are computed for each pair and aggregated alongside predicted sex probabilities when provided.
