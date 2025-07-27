# dimensionality_reduction_wrapper.py

## Überblick
Transformer, der PCA, UMAP, t-SNE, LDA, MDS und Isomap bereitstellt.
Die Klasse ermöglicht einen einheitlichen Aufruf verschiedener Reduktionsverfahren. Sie wird eingesetzt, um hochdimensionale Merkmalsräume für Visualisierung oder Klassifikation vorzubereiten. Vorteilhaft ist die flexible Wahl des Verfahrens, während einige Methoden wie t-SNE rechenintensiv sein können.

## Wichtige Bestandteile
- DimensionalityReducerTransformer

## Referenzen
- https://scikit-learn.org/stable/modules/decomposition.html
- https://umap-learn.readthedocs.io/en/latest/

## Annahmen und Einschränkungen
Wählt die Methode anhand der Parameter; UMAP wird stets beaufsichtigt trainiert, sodass die Option `supervised` nur andere Verfahren betrifft.
