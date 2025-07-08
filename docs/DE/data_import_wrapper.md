# data_import_wrapper.py

## Überblick
Fasst Import, Bereinigung und Konvertierung aller Rohdatensätze zusammen.

Der Wrapper kombiniert die Hilfsfunktionen zu einem konsistenten Ablauf. Eingelesene Daten werden vereinheitlicht, fehlende Werte behandelt und anschließend in Parquet-Dateien gespeichert. So lassen sich Datenpipelines reproduzierbar gestalten.
## Wichtige Bestandteile
- DataImporter
- DataImportWrapper

## Referenzen
- https://pandas.pydata.org/docs/

## Annahmen und Einschränkungen
Ausgelegt für Parquet-Ausgabe und numerische Konvertierung vor dem Skalieren.
