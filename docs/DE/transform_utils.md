# transform_utils.py

## Überblick
Hilfsfunktionen zum Konvertieren von Feature-Spalten und Kodieren der Zielvariablen.
"convert_numeric" wandelt Spalten in numerische Werte um und führt bei rein textuellen Merkmalen automatisch ein One-Hot-Encoding durch. "one_hot_encode_targets" erzeugt Dummy-Variablen für Zielgrößen und "save_target_mapping" speichert die Zuordnung als JSON. Sie sind besonders in der Vorverarbeitung für Machine-Learning-Pipelines nützlich.

## Wichtige Bestandteile
- convert_numeric
- one_hot_encode_targets
- save_target_mapping

## Referenzen
- https://pandas.pydata.org/docs/

## Annahmen und Einschränkungen
Unterstützt Kommazahlen und JSON-Mapping-Dateien.
