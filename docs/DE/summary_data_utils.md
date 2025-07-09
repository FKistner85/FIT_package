# summary_data_utils.py

## Überblick
Funktionen zur Zusammenfassung der Split-Statistiken und zum Erzeugen von Balkendiagrammen.
Die Gestaltung der Grafiken erfolgt über `FIT_python.plot_style.apply_style()`.
Mit diesen Werkzeugen lassen sich die Ergebnisse der Datenaufteilung übersichtlich darstellen. Sie erzeugen Tabellen und Abbildungen, die zur Qualitätssicherung herangezogen werden können. Voraussetzung ist eine konsistente Spaltenstruktur.

## Wichtige Bestandteile
- compute_summary
- plot_summary_table
- plot_split_proportions

### plot_split_proportions
Zeigt den relativen Anteil der Footprints pro Split (Train, Test, Inference)
je Art. Ergänzt damit die absoluten Zahlen aus `plot_summary_table`.

## Referenzen
- https://pandas.pydata.org/docs/
- https://matplotlib.org/

## Annahmen und Einschränkungen
Erwartet Spalten wie "sex" und "individual_id".
