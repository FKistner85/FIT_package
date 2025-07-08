# summary_data_wrapper.py

## Überblick
Erzeugt Zusammenfassungen für alle Split-Dateien und erstellt optional Plots.
Die Klasse fasst die Split-Ausgaben verschiedener Arten zusammen und erzeugt daraus eine Gesamtübersicht. Neben CSV-Dateien können auch Grafiken erstellt werden. Dies erleichtert die Dokumentation, führt aber zu zusätzlicher Rechenzeit.

## Wichtige Bestandteile
- run_summary
- SummaryWrapper.summarize_all

## Referenzen
- https://pandas.pydata.org/docs/

## Annahmen und Einschränkungen
Schreibt Zusammenfassungen und Abbildungen im results-Verzeichnis.
