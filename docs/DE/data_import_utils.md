# data_import_utils.py

## Überblick
Hilfsfunktionen zum Laden von CSV-/Excel-Dateien und zur Normalisierung der Spaltennamen.

Die Funktionen unterstützen beim Vorverarbeiten heterogener Rohdaten. "clean_columns" vereinheitlicht Spaltennamen, "load_raw_files" liest ganze Ordner ein und fügt optional IDs hinzu. "coerce_numeric_columns" konvertiert Textspalten zu numerischen Werten, während "sanitize_labels" Beschriftungen bereinigt.
## Wichtige Bestandteile
- clean_columns
- load_raw_files
- coerce_numeric_columns
- sanitize_labels

### clean_columns
Normalisiert Spaltennamen, indem sie in Kleinbuchstaben umgewandelt, Leerzeichen entfernt und Nicht-Wort-Zeichen durch Unterstriche ersetzt werden. Diese Routine sollte direkt nach dem Einlesen der Rohdaten laufen, damit nachfolgender Code auf einheitliche Bezeichnungen vertrauen kann. Bei sehr ähnlichen Spalten kann es zu doppelten Labels kommen.

### load_raw_files
Liest alle CSV- oder Excel-Dateien in einem Verzeichnis ein, wendet `clean_columns` an und fügt bei Bedarf eine `id`-Spalte ein. Alle zurückgegebenen DataFrames teilen dadurch dasselbe Namensschema. Nicht unterstützte Dateiendungen werden übersprungen, was bei stark unterschiedlichen Layouts leicht zu übersehen ist.

### coerce_numeric_columns
Durchsucht Textspalten nach Werten, die wie Zahlen mit Komma als Dezimaltrennzeichen aussehen, und wandelt sie in `float` um. Dieser Schritt sollte vor Skalierung oder Modellierung erfolgen, damit numerische Operationen korrekt arbeiten. Tausendertrennzeichen werden nicht erkannt und können zu falsch interpretierten Werten führen.

## Referenzen
- https://pandas.pydata.org/docs/
- https://scikit-learn.org/stable/modules/preprocessing.html

## Annahmen und Einschränkungen
Geht von überwiegend numerischen Daten mit optionalen Kommazahlen aus.
