# pairwise_individual_id_pipeline.py

## Überblick
Hauptpipeline zur Berechnung paarweiser Distanzen nach Feature-Selektion und Reduktion.
Die Pipeline kombiniert Feature-Selektionsmethoden, Dimensionsreduktion und Distanzberechnung. Sie kommt bei Identitätsprüfungen zum Einsatz und kann optional Wahrscheinlichkeiten aus einem Sexmodell berücksichtigen. Die Modularität erleichtert Experimente, führt bei vielen Paaren jedoch zu hohem Speicherbedarf.

## Wichtige Bestandteile
- run_all_pairwise_projections_parallel
- run_embedding_once_pipeline

### run_embedding_once_pipeline
Berechnet die Trainings-Trails nur einmal und transformiert danach je Vergleich
nur die Test-Spur. Hilfreich, wenn die Referenz öfter wiederverwendet wird.

## Referenzen
- https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html

## Annahmen und Einschränkungen
Optional kann die Pipeline Sexmodell-Wahrscheinlichkeiten hinzufügen.
