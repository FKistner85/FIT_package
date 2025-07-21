# distance_metrics.py

## Überblick
Stellt `compute_distances` bereit und liefert diverse Distanzmaße.
Die Funktion berechnet mehrere Distanzwerte gleichzeitig und gibt sie als Dictionary zurück. Sie eignet sich zum Vergleich geometrischer Repräsentationen oder als Grundlage für Clustering. Der Ansatz ist kompakt, unterstützt jedoch keine Batch-Verarbeitung.

## Wichtige Bestandteile
- compute_distances

## Referenzen
- https://docs.scipy.org/doc/scipy/reference/spatial.distance.html

## Annahmen und Einschränkungen
Die Funktion gibt die aufgeführten Distanzen nur für einzelne Vektoren zurück
und unterstützt keine Batch-Verarbeitung.
