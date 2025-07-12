# feature_corr_utils.py

## Überblick
Hilfsfunktion zur Visualisierung der Korrelation zwischen Feature-Gruppen.

## Wichtige Bestandteile
- `plot_feature_correlations`

### plot_feature_correlations
Ermittelt eine Korrelationsmatrix basierend auf aggregierten
Feature-Gruppen und speichert eine Heatmap. Spalten mit `d` oder
`dist` werden der Gruppe *distance* zugeordnet, `ang` der Gruppe
*angle* und `t` (ohne `trail`) der Gruppe *triangles*. Sind weniger
als zwei Gruppen vorhanden, wird die Heatmap für alle numerischen
Features ohne Achsenbeschriftung erzeugt.
