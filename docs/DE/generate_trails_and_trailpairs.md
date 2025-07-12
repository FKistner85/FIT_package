# generate_trails_and_trailpairs.py (veraltet)

## Überblick
Erzeugte Trail-Segmente und alle Paarvergleiche mit Metadaten.
Die Funktionalität befindet sich nun in `geometric_pairwise_projection.generate_pairwise_comparisons_from_df`.

## Wichtige Bestandteile
- generate_pairwise_comparisons_from_df (verschoben nach `geometric_pairwise_projection`)

## Referenzen
- https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html

## Annahmen und Einschränkungen
Trails können direkt vorgegeben oder über zufällige Teilmengen mit Jaccard-basierter Auswahl erzeugt werden; die Ergebnisse enthalten Zusammenfassungsstatistiken.
