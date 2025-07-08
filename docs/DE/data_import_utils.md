# data_import_utils.py

## Überblick
Hilfsfunktionen zum Laden von CSV-/Excel-Dateien und zur Normalisierung der Spaltennamen.

Die Funktionen unterstützen beim Vorverarbeiten heterogener Rohdaten. "clean_columns" vereinheitlicht Spaltennamen, "load_raw_files" liest ganze Ordner ein und fügt optional IDs hinzu. "coerce_numeric_columns" konvertiert Textspalten zu numerischen Werten, während "sanitize_labels" Beschriftungen bereinigt.
## Wichtige Bestandteile
- clean_columns
- load_raw_files
- coerce_numeric_columns
- sanitize_labels

## Referenzen
- https://pandas.pydata.org/docs/
- https://scikit-learn.org/stable/modules/preprocessing.html

## Annahmen und Einschränkungen
Geht von überwiegend numerischen Daten mit optionalen Kommazahlen aus.
