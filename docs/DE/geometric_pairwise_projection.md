# geometric_pairwise_projection.py

## Überblick
Führt paarweise Projektionen mit optionalen Sex-Modell-Wahrscheinlichkeiten aus und cached die Resultate.

## Wichtige Bestandteile
- generate_pairwise_comparisons_from_df
- run_all_pairwise_projections_parallel

## Referenzen
- https://joblib.readthedocs.io/
- https://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.LinearDiscriminantAnalysis.html

## Annahmen und Einschränkungen
Processes comparisons in batches; uses dimensionality reduction before distance computation.
