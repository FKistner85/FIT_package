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

* **PCA** eignet sich für schnelle lineare Projektionen und ist gut interpretierbar. Die Methode ist sinnvoll, wenn eine lineare Struktur vorliegt und Varianz ein guter Indikator für Information ist. Nachteilig ist, dass nur lineare Zusammenhänge abgebildet werden und eine Skalierung der Eingaben meist notwendig ist.

* **UMAP** bewahrt lokale Nachbarschaften auch bei nichtlinearen Datenstrukturen. Es eignet sich vor allem für Visualisierungen mit wenigen Dimensionen und kann sowohl unbeaufsichtigt als auch mit Klasseninformation eingesetzt werden. Die Wahl der Nachbarschaftsgröße beeinflusst das Ergebnis stark und globale Distanzen können verzerrt werden.

* **t-SNE** ist besonders für explorative Cluster-Analysen beliebt. Die stochastische Einbettung erzeugt anschauliche zwei- oder dreidimensionale Karten, lässt sich aber schwer auf neue Datenpunkte übertragen und ist rechenintensiv.

* **LDA** nutzt Klassenlabels, um Achsen mit maximaler Trennschärfe zu finden. Vorausgesetzt werden annähernd normalverteilte Klassen mit gleichen Kovarianzmatrizen. Bei starken Abweichungen von diesen Annahmen sinkt die Qualität der Projektion.

* **MDS** versucht die Paarabstände aus dem ursprünglichen Raum zu bewahren. Das Verfahren verschafft einen Einblick in die intrinsische Struktur, skaliert aber schlecht mit vielen Beispielen.

* **Isomap** erweitert MDS um die Approximation geodätischer Distanzen auf einem Nachbarschaftsgraphen. Es kann gekrümmte Mannigfaltigkeiten entfalten, benötigt jedoch eine ausreichend dichte Stichprobe.

### Referenzen
* Fisher, R. A. (1936). "The use of multiple measurements in taxonomic problems." *Annals of Eugenics*.
* Hotelling, H. (1933). "Analysis of a complex of statistical variables into principal components." *Journal of Educational Psychology*.
* Kruskal, J. B. (1964). "Multidimensional scaling by optimizing goodness of fit to a nonmetric hypothesis." *Psychometrika*.
* McInnes, L., Healy, J., & Melville, J. (2018). "UMAP: Uniform Manifold Approximation and Projection for dimension reduction." arXiv:1802.03426.
* Pearson, K. (1901). "On lines and planes of closest fit to systems of points in space." *Philosophical Magazine*.
* Tenenbaum, J. B., de Silva, V., & Langford, J. C. (2000). "A global geometric framework for nonlinear dimensionality reduction." *Science*.
* van der Maaten, L., & Hinton, G. (2008). "Visualizing data using t-SNE." *Journal of Machine Learning Research*.
