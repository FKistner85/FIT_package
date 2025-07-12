# pairwise_individual_id_pipeline.py

## Überblick
Hauptpipeline zur Berechnung paarweiser Distanzen nach Feature-Selektion und Reduktion.
Die Pipeline kombiniert Feature-Selektionsmethoden, Dimensionsreduktion und Distanzberechnung. Sie kommt bei Identitätsprüfungen zum Einsatz und kann optional Wahrscheinlichkeiten aus einem Sexmodell berücksichtigen. Die Modularität erleichtert Experimente, führt bei vielen Paaren jedoch zu hohem Speicherbedarf.

## Wichtige Bestandteile
- run_all_pairwise_projections_parallel

### run_all_pairwise_projections_parallel
Übernimmt die vollständige Verarbeitung aller Trail-Vergleiche. Die Funktion führt Ausreißerbehandlung, Skalierung, Feature-Selektion und Dimensionsreduktion durch, bevor mehrere Distanzmetriken berechnet werden. Übergeben wird eine Liste von Vergleichen aus `generate_trails_and_trailpairs` sowie das Basis-DataFrame. Umfangreiche Parameterkombinationen können lange Laufzeiten verursachen, daher empfiehlt sich eine parallele Ausführung über `n_jobs`.

## Referenzen
- https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html

## Annahmen und Einschränkungen
Optional kann die Pipeline Sexmodell-Wahrscheinlichkeiten hinzufügen.
