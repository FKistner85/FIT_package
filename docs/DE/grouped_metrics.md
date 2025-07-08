# grouped_metrics.py

## Überblick
Funktionen zur Berechnung individueller Genauigkeitswerte.
Mit diesen Routinen lassen sich Vorhersagen pro Individuum zusammenfassen. Sie liefern Accuracy-Werte und Mehrheitsentscheide für Gruppen. Voraussetzung ist eine korrekte Zuordnung von Labels.

## Wichtige Bestandteile
- individual_accuracies
- individual_majority_stats

## Referenzen
- https://pandas.pydata.org/docs/

## Annahmen und Einschränkungen
Gruppenlabels müssen zusammen mit den Vorhersagen vorliegen.
