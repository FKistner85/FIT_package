# transform_utils.py

## Überblick
Hilfsfunktionen zum Konvertieren von Feature-Spalten und Kodieren der Zielvariablen.
Die Funktionen vereinfachen die Datenvorbereitung: "convert_numeric" erkennt Kommazahlen, "one_hot_encode_targets" erzeugt Dummy-Variablen und "save_target_mapping" speichert die Zuordnung als JSON. Sie sind besonders in der Vorverarbeitung für Machine-Learning-Pipelines nützlich.

## Wichtige Bestandteile
- convert_numeric
- one_hot_encode_targets
- save_target_mapping

## Referenzen
- https://pandas.pydata.org/docs/

## Annahmen und Einschränkungen
Unterstützt Kommazahlen und JSON-Mapping-Dateien.
