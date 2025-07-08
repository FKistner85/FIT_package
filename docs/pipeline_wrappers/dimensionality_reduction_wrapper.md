# dimensionality_reduction_wrapper.py

Die Klasse `DimensionalityReducerTransformer` stellt mehrere Verfahren bereit:

```python
    Wrapper für Dimensionsreduktion:
      - None: Identity
      - PCA
      - UMAP
      - t-SNE
      - LDA
      - MDS
      - Isomap
```
【F:src/FIT_python/pipeline_sex/dimensionality_reduction_wrapper.py†L12-L21】

*PCA* liefert lineare Hauptachsen und ist leicht interpretierbar, *UMAP* und *t-SNE* bewahren nichtlineare Nachbarschaften für Visualisierung, während *LDA* auf Klasseninformation basiert. *MDS* und *Isomap* versuchen Distanzen zu erhalten, benötigen aber oft mehr Rechenzeit.
