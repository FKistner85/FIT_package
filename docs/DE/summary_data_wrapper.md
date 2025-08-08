# summary_data_wrapper.py

## Überblick
Erzeugt Zusammenfassungen für Split-Dateien einer Art und erstellt optional Plots.
Die Ausgabepfade werden über `get_species_paths(section="dataprocessing", species)` bestimmt, sodass Tabellen und Grafiken in den jeweiligen Unterordnern der Art landen. Dies erleichtert die Dokumentation, führt aber zu zusätzlicher Rechenzeit.

## Wichtige Bestandteile
- run_summary: erstellt pro Art eine Parquet-Tabelle und Plots in den Artspezifischen Verzeichnissen.
- SummaryWrapper.summarize_all: iteriert über alle Arten und ruft `run_summary` für jede auf.

## Referenzen
- https://pandas.pydata.org/docs/

## Annahmen und Einschränkungen
Schreibt Zusammenfassungen und Abbildungen in artspezifische Ordner unterhalb des results-Verzeichnisses.
