# generate_trails_and_trailpairs.py

## Überblick
Erzeugt Trail-Segmente und alle Paarvergleiche mit Metadaten.
Das Modul dient dazu, Bewegungsabläufe in vergleichbare Abschnitte zu zerlegen und daraus Paarungen zu bilden. Dadurch lassen sich individuelle Muster miteinander kontrastieren. Bei großen Datensätzen können die Berechnungen jedoch zeitaufwändig werden.

## Wichtige Bestandteile
- generate_pairwise_comparisons_from_df

## Referenzen
- https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html

## Annahmen und Einschränkungen
Trails können direkt vorgegeben oder per Sliding-Window zufällig ausgewählt werden; die Ergebnisse enthalten Zusammenfassungsstatistiken.
