# geometric_pairwise_projection.py

## Überblick
Führt paarweise Projektionen mit optionalen Sex-Modell-Wahrscheinlichkeiten aus und cached die Resultate.
Die Funktion projiziert die Merkmale in einen reduzierten Raum und berechnet anschließend Distanzen zwischen den Individuen. Zwischenergebnisse werden per Joblib gespeichert, was parallele Ausführung ermöglicht. Hohe Geschwindigkeit ist ein Vorteil, während der Speicherbedarf anwachsen kann.

## Wichtige Bestandteile
- generate_pairwise_comparisons_from_df
- run_all_pairwise_projections_parallel

## Referenzen
- https://joblib.readthedocs.io/
- https://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.LinearDiscriminantAnalysis.html

## Annahmen und Einschränkungen
Vergleicht die Paare stapelweise und führt vor der Distanzberechnung eine Dimensionsreduktion durch.
